/*
    hip/accel.cpp -- HIP-RT acceleration structure builder.

    Consumes the backend-neutral SceneIR (scene_ir.h) and produces the HIP-RT
    objects that jit_hip_ray_trace() traverses. The Metal counterpart is
    src/render/metal_accel.mm; the OptiX one is the top half of
    src/render/scene_optix.inl.

    Three things here are specific to HIP-RT rather than transliterated from
    those, and each is load-bearing:

    1. INSTANCES ARE EXPANDED. A hiprtInstance references exactly one geometry
       and a hiprtHit reports no geometry ID (verified in hiprt_types.h: the hit
       is {instanceID, primID, uv, normal, t}). A SceneIR BlasEntry holding N
       same-kind geometries therefore cannot map to one instance. Each
       (InstanceEntry, geometry) pair becomes its own HIP-RT instance, and the
       geometry index is handed back to the kernel through the instance-indexed
       table that jit_hip_configure_scene() takes. By the time Mitsuba sees the
       values they are exactly Metal's, which is what lets scene_hip.inl reuse
       scene_metal.inl's recovery logic unchanged.

    2. THE CONTEXT IS BORROWED, NEVER CREATED. hiprtGeometry and hiprtScene
       handles are context-scoped, and drjit-core compiles its traversing
       kernels against its own context. Building against a second context does
       not fail -- it traverses garbage. jit_hip_rt_context() exists so there is
       one context, as jit_metal_context() does for Metal.

    3. DEVICE MEMORY COMES FROM DRJIT, NOT hipMalloc. Under the CUDA shim the
       "HIP" backend is CUDA-backed, so every allocation has to go through
       jit_malloc(JitBackend::HIP, ...), which allocates on whichever runtime is
       actually in play. Calling hipMalloc here would work on the MI210 and fail
       on the development machine (BACKEND_NOTES 11f).
*/

#if defined(MI_ENABLE_HIP)

#include "accel.h"

#include <drjit-core/hip.h>
#include <drjit-core/jit.h>
#include <mitsuba/core/logger.h>

#include <hiprt/hiprt.h>

#include <cstring>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

/// Owns every HIP-RT object belonging to one built scene.
struct HIPAccelData {
    hiprtContext context = nullptr;
    hiprtScene scene = nullptr;
    /// One per expanded (instance, geometry) pair, in instance order.
    std::vector<hiprtGeometry> geometries;
    /// Device buffers handed to HIP-RT or to jit_hip_configure_scene().
    std::vector<void *> device_buffers;
};

// ---------------------------------------------------------------------------
//  Small helpers
// ---------------------------------------------------------------------------

#define HIPRT_CHECK(expr)                                                      \
    do {                                                                       \
        hiprtError rv_ = (expr);                                               \
        if (rv_ != hiprtSuccess)                                               \
            Throw("build_hip_accel(): %s failed (%i).", #expr, (int) rv_);     \
    } while (0)

/// Allocate device memory owned by \c d and copy \c size bytes of \c src into it.
static void *upload(HIPAccelData *d, const void *src, size_t size) {
    if (size == 0)
        return nullptr;
    void *ptr = jit_malloc(JitBackend::HIP, size);
    jit_memcpy(JitBackend::HIP, ptr, src, size);
    d->device_buffers.push_back(ptr);
    return ptr;
}

/// Is this geometry kind one HIP-RT traverses as a triangle mesh?
static bool is_triangle_kind(ShapeIR::Kind k) {
    return k == ShapeIR::Kind::Triangles || k == ShapeIR::Kind::TrianglesCulled;
}

// ---------------------------------------------------------------------------
//  Geometry construction
// ---------------------------------------------------------------------------

/// Fill \c bi for one ShapeIR, uploading whatever the shape needs.
///
/// Returns false for a geometry kind this backend cannot build yet, having
/// already reported why. The caller must not build it.
static bool make_build_input(HIPAccelData *d, const ShapeIR &g,
                             ShapeIR::Kind kind, hiprtGeometryBuildInput &bi) {
    bi = {};

    if (is_triangle_kind(kind)) {
        // vertex_ptr / index_ptr are already DEVICE pointers on a GPU backend
        // (see ShapeIR), so there is nothing to copy.
        bi.type = hiprtPrimitiveTypeTriangleMesh;
        bi.primitive.triangleMesh.vertices = (hiprtDevicePtr) g.vertex_ptr;
        bi.primitive.triangleMesh.vertexCount = (uint32_t) g.vertex_count;
        bi.primitive.triangleMesh.vertexStride = 3 * sizeof(float);
        bi.primitive.triangleMesh.triangleIndices = (hiprtDevicePtr) g.index_ptr;
        bi.primitive.triangleMesh.triangleCount = (uint32_t) g.face_count;
        bi.primitive.triangleMesh.triangleStride = 3 * sizeof(uint32_t);
        return true;
    }

    if (kind == ShapeIR::Kind::Custom) {
        // The AABBs themselves are easy -- fill_aabbs() writes prim_count * 6
        // floats. What is missing is the other half: an AABB-list geometry only
        // reports a hit if a custom intersection function says so, and wiring
        // Mitsuba's per-ShapeType intersectors into a hiprtFuncTable is not
        // done yet.
        //
        // Refusing loudly rather than building it. A geometry built here with
        // no func table traverses to the stub intersectFunc, which reports no
        // hit -- so every sphere, disk, cylinder and sdfgrid in the scene would
        // silently render as empty space. This project has had enough failures
        // that look like a working system.
        Throw("build_hip_accel(): the scene contains custom (implicit) "
              "geometry -- sphere, disk, cylinder, sdfgrid or ellipsoids -- "
              "which the HIP backend cannot intersect yet: its hiprtFuncTable "
              "intersection callbacks are not implemented. Triangle meshes are "
              "supported. Use a mesh-only scene, or the llvm/cuda variants.");
    }

    if (kind == ShapeIR::Kind::BSplineCurve || kind == ShapeIR::Kind::LinearCurve)
        Throw("build_hip_accel(): the scene contains curve geometry, which "
              "the HIP backend does not support yet. Triangle meshes are "
              "supported.");

    Throw("build_hip_accel(): unhandled geometry kind %i.", (int) kind);
}

/// Build one hiprtGeometry from \c bi.
static hiprtGeometry build_geometry(HIPAccelData *d,
                                    const hiprtGeometryBuildInput &bi,
                                    bool compact) {
    hiprtBuildOptions opts{};
    opts.buildFlags = compact ? hiprtBuildFlagBitPreferHighQualityBuild
                              : hiprtBuildFlagBitPreferBalancedBuild;

    size_t temp_size = 0;
    HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(d->context, bi, opts,
                                                         temp_size));

    void *temp = nullptr;
    if (temp_size)
        temp = jit_malloc(JitBackend::HIP, temp_size);

    hiprtGeometry geom = nullptr;
    HIPRT_CHECK(hiprtCreateGeometry(d->context, bi, opts, geom));
    HIPRT_CHECK(hiprtBuildGeometry(d->context, hiprtBuildOperationBuild, bi,
                                   opts, (hiprtDevicePtr) temp,
                                   /* stream = */ nullptr, geom));

    // The build ran on the null stream while Dr.Jit works on its own; sync
    // before the scratch buffer is recycled or the scene build reads the
    // result. This is one-time setup, so a full sync costs nothing that
    // matters.
    jit_sync_thread();

    if (temp)
        jit_free(temp);

    return geom;
}

// ---------------------------------------------------------------------------
//  Scene construction
// ---------------------------------------------------------------------------

std::pair<HIPAccelData *, uint32_t> build_hip_accel(const SceneIR &sd,
                                                    bool compact) {
    hiprtContext context = (hiprtContext) jit_hip_rt_context();
    if (!context)
        Throw("build_hip_accel(): no HIP-RT context. Dr.Jit was built without "
              "a usable HIP-RT, so ray tracing is unavailable on this backend.");

    HIPAccelData *d = new HIPAccelData();
    d->context = context;

    // --- Expand (instance, geometry) pairs into HIP-RT instances -----------
    //
    // See note 1 at the top of this file. The three arrays below are all
    // indexed by the expanded instance id, which is what a hiprtHit reports.
    std::vector<hiprtInstance> instances;
    std::vector<hiprtFrameMatrix> frames;
    std::vector<uint32_t> geometry_ids, user_instance_ids;

    bool any_backface_culled = false;

    for (const InstanceEntry &inst : sd.instances) {
        const BlasEntry &blas = sd.blases[inst.blas_index];
        if (blas.kind == ShapeIR::Kind::TrianglesCulled)
            any_backface_culled = true;

        for (uint32_t geom_idx = 0; geom_idx < blas.geoms.size(); ++geom_idx) {
            hiprtGeometryBuildInput bi;
            if (!make_build_input(d, blas.geoms[geom_idx], blas.kind, bi))
                continue;

            hiprtGeometry geom = build_geometry(d, bi, compact);
            d->geometries.push_back(geom);

            hiprtInstance hi{};
            hi.type = hiprtInstanceTypeGeometry;
            hi.geometry = geom;
            instances.push_back(hi);

            // SceneIR stores to_world column-major as four columns of three
            // floats, i.e. element (row, col) is to_world[col * 3 + row].
            // hiprtFrameMatrix is matrix[row][col]. Transposing wrongly here
            // would place geometry plausibly but incorrectly, so it is spelled
            // out rather than memcpy'd.
            hiprtFrameMatrix fm{};
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col)
                    fm.matrix[row][col] = inst.to_world[col * 3 + row];
            fm.time = 0.f;
            frames.push_back(fm);

            geometry_ids.push_back(geom_idx);
            user_instance_ids.push_back(
                inst.owner_registry_id == SCENE_IR_NO_OWNER
                    ? 0u
                    : inst.owner_registry_id);
        }
    }

    if (instances.empty()) {
        delete d;
        Throw("build_hip_accel(): the scene lowered to no buildable geometry.");
    }

    // One frame per instance, so the transform headers are 1:1.
    std::vector<hiprtTransformHeader> headers(instances.size());
    for (uint32_t i = 0; i < instances.size(); ++i)
        headers[i] = { /* frameIndex = */ i, /* frameCount = */ 1u };

    // --- Build the scene ---------------------------------------------------

    hiprtSceneBuildInput sbi{};
    sbi.instances = (hiprtDevicePtr) upload(
        d, instances.data(), instances.size() * sizeof(hiprtInstance));
    sbi.instanceTransformHeaders = (hiprtDevicePtr) upload(
        d, headers.data(), headers.size() * sizeof(hiprtTransformHeader));
    sbi.instanceFrames = (hiprtDevicePtr) upload(
        d, frames.data(), frames.size() * sizeof(hiprtFrameMatrix));
    sbi.instanceMasks = nullptr; // NULL => hiprtFullRayMask for every instance
    sbi.instanceCount = (uint32_t) instances.size();
    sbi.frameCount = (uint32_t) frames.size();
    sbi.frameType = hiprtFrameTypeMatrix;

    hiprtBuildOptions opts{};
    opts.buildFlags = compact ? hiprtBuildFlagBitPreferHighQualityBuild
                              : hiprtBuildFlagBitPreferBalancedBuild;

    size_t temp_size = 0;
    HIPRT_CHECK(
        hiprtGetSceneBuildTemporaryBufferSize(context, sbi, opts, temp_size));

    void *temp = nullptr;
    if (temp_size)
        temp = jit_malloc(JitBackend::HIP, temp_size);

    HIPRT_CHECK(hiprtCreateScene(context, sbi, opts, d->scene));
    HIPRT_CHECK(hiprtBuildScene(context, hiprtBuildOperationBuild, sbi, opts,
                                (hiprtDevicePtr) temp, /* stream = */ nullptr,
                                d->scene));
    jit_sync_thread();

    if (temp)
        jit_free(temp);

    // --- Register with Dr.Jit ----------------------------------------------

    void *geometry_ids_dev = upload(d, geometry_ids.data(),
                                    geometry_ids.size() * sizeof(uint32_t));
    void *user_ids_dev = upload(d, user_instance_ids.data(),
                                user_instance_ids.size() * sizeof(uint32_t));

    // Bit 0 triangles, bit 1 custom/AABB, bit 2 curves, bit 3 backface culling
    // required -- the same encoding Metal uses. Only triangles can be built
    // above, so bits 1 and 2 are unreachable for now; the mask is assembled
    // from what was actually built rather than hardcoded, so it stays honest
    // when the custom path lands.
    uint32_t types_mask = 1u; // triangles
    if (any_backface_culled)
        types_mask |= 1u << 3;

    uint32_t scene_index = jit_hip_configure_scene(
        (void *) d->scene, /* func_table = */ nullptr, geometry_ids_dev,
        user_ids_dev, types_mask);

    // The HIP-RT objects must outlive the scene variable, which can outlast
    // this scope: unevaluated kernels and frozen recordings reference it
    // through their TraceRay nodes. Freeing happens when that reference count
    // hits zero, not when release_hip_accel() is called.
    jit_hip_scene_set_cleanup(
        scene_index,
        [](void *payload) {
            HIPAccelData *dd = (HIPAccelData *) payload;
            for (hiprtGeometry g : dd->geometries)
                hiprtDestroyGeometry(dd->context, g);
            if (dd->scene)
                hiprtDestroyScene(dd->context, dd->scene);
            for (void *p : dd->device_buffers)
                jit_free(p);
            delete dd;
        },
        (void *) d);

    Log(Debug,
        "build_hip_accel(): built %zu HIP-RT instances over %zu geometries.",
        instances.size(), d->geometries.size());

    return { d, scene_index };
}

void release_hip_accel(HIPAccelData * /*accel*/, uint32_t scene_index) {
    // Drop this owner's reference. The cleanup callback registered above frees
    // the HIP-RT objects (and the HIPAccelData) once no kernel or recording
    // still references the scene, so nothing is deleted here.
    jit_var_dec_ref(scene_index);
}

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_HIP
