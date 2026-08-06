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
#include "shapes.h"

#include <drjit-core/hip.h>
#include <drjit-core/jit.h>
#include <mitsuba/core/logger.h>

#include <hiprt/hiprt.h>

#include <cstring>
#include <string>
#include <vector>

/// hip/intersection_functions.hip, embedded verbatim by the build
/// (src/render/CMakeLists.txt). Not NUL-terminated -- bin2c emits raw bytes.
extern "C" const char mi_hip_isect_source[];
extern "C" const size_t mi_hip_isect_source_size;

NAMESPACE_BEGIN(mitsuba)

/// Owns every HIP-RT object belonging to one built scene.
struct HIPAccelData {
    hiprtContext context = nullptr;
    hiprtScene scene = nullptr;
    /// Custom-primitive dispatch table, or null for a mesh-only scene.
    hiprtFuncTable func_table = nullptr;
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

/// Register the custom-primitive device source with drjit-core.
///
/// The source is not linked into a library the way Metal's .metallib is:
/// hiprtBuildTraceKernels() generates the intersectFunc dispatcher from
/// funcNameSets and compiles it together with the kernel, so drjit-core has to
/// prepend this text to EVERY traversing kernel it emits
/// (BACKEND_NOTES 7a-3, 11o). Registration is global and idempotent; calling it
/// once per scene build is fine and keeps mesh-only processes from paying for
/// intersection code they will never dispatch to.
static void register_isect_source() {
    static const std::string source(mi_hip_isect_source,
                                    mi_hip_isect_source_size);

    // No filter functions: Mitsuba's any-hit logic is in the integrator, and a
    // filter that always accepts is exactly what HIP-RT does with a null entry.
    const char *filters[HIP_ISECT_FN_COUNT] = {};

    jit_hip_set_isect_source(source.c_str(), hip_isect_fn_names, filters,
                             HIP_ISECT_FN_COUNT);
}

// ---------------------------------------------------------------------------
//  Geometry construction
// ---------------------------------------------------------------------------

/// Per-geometry-type accumulator for custom-primitive records.
///
/// One combined array per type, because a HIP-RT function table hands the
/// intersection function ONE data pointer per geometry type (hip/shapes.h).
/// Each geometry's slice is found through the instance-indexed \c base table.
struct CustomTypeData {
    /// Concatenated per-primitive records, host side.
    std::vector<uint8_t> records;
    /// Per-primitive record size; all shapes of one type must agree.
    size_t elem_size = 0;
};

/// Fill \c bi for one ShapeIR, uploading whatever the shape needs.
///
/// For custom (AABB) geometry this also appends the shape's per-primitive
/// records to \c types[fn] and reports the element index they start at through
/// \c base_out, which the device reads via \c HIPIsectTypeData::base.
static bool make_build_input(HIPAccelData *d, const ShapeIR &g,
                             ShapeIR::Kind kind, hiprtGeometryBuildInput &bi,
                             CustomTypeData *types, uint32_t &base_out) {
    bi = {};
    base_out = 0;

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
        uint32_t fn = hip_fn_index(g.type);

        // Refuse loudly rather than build it. An AABB geometry whose type has
        // no intersection function dispatches to a `default: return false` arm
        // of HIP-RT's generated switch, which reports a miss for every ray --
        // so the shape would render as empty space with nothing anywhere
        // reporting a problem. This project has had enough failures that look
        // like a working system.
        if (fn >= HIP_ISECT_FN_COUNT)
            Throw("build_hip_accel(): the scene contains a custom (implicit) "
                  "shape of type 0x%x, which the HIP backend cannot intersect "
                  "yet -- only spheres and triangle meshes are implemented. "
                  "Use the llvm or cuda variants for scenes with disks, "
                  "cylinders, sdfgrids or ellipsoids.",
                  (uint32_t) g.type);

        if (g.prim_count == 0 || g.pdata_size == 0 || !g.fill_aabbs ||
            !g.fill_data)
            Throw("build_hip_accel(): custom geometry of type 0x%x described "
                  "itself incompletely (%zu primitives, %zu bytes each, "
                  "fill_aabbs %s, fill_data %s). A missing callback usually "
                  "means the shape's describe() is behind a preprocessor gate "
                  "that does not list HIP -- see MI_GPU_CUSTOM_SHAPES.",
                  (uint32_t) g.type, g.prim_count, g.pdata_size,
                  g.fill_aabbs ? "ok" : "MISSING",
                  g.fill_data ? "ok" : "MISSING");

        // --- AABBs -------------------------------------------------------
        // One device buffer per geometry rather than a shared suballocated
        // pool: HIP-RT indexes them with a byte stride from the base pointer,
        // and separate allocations keep every base 16-byte aligned without
        // padding arithmetic. Custom geometries are few.
        std::vector<float> aabbs(g.prim_count * 6);
        g.fill_aabbs(g.ctx, aabbs.data());
        void *aabb_dev = upload(d, aabbs.data(), aabbs.size() * sizeof(float));

        // --- Per-primitive records ---------------------------------------
        CustomTypeData &td = types[fn];
        if (td.elem_size != 0 && td.elem_size != g.pdata_size)
            Throw("build_hip_accel(): shapes of type 0x%x disagree on their "
                  "per-primitive data size (%zu vs %zu). The combined buffer "
                  "strides by a single element size.",
                  (uint32_t) g.type, td.elem_size, g.pdata_size);
        td.elem_size = g.pdata_size;

        size_t total = g.data_size_bytes();
        base_out = (uint32_t) (td.records.size() / g.pdata_size);
        size_t off = td.records.size();
        td.records.resize(off + total);
        g.fill_data(g.ctx, td.records.data() + off);

        bi.type = hiprtPrimitiveTypeAABBList;
        bi.geomType = fn;
        bi.primitive.aabbList.aabbs = (hiprtDevicePtr) aabb_dev;
        bi.primitive.aabbList.aabbCount = (uint32_t) g.prim_count;
        bi.primitive.aabbList.aabbStride = 6 * sizeof(float);
        return true;
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

    // Custom-primitive records, accumulated per geometry type, plus the
    // instance-indexed base table the device uses to find its slice.
    CustomTypeData custom_types[HIP_ISECT_FN_COUNT];
    std::vector<uint32_t> custom_base;

    bool any_backface_culled = false, any_custom = false;

    for (const InstanceEntry &inst : sd.instances) {
        const BlasEntry &blas = sd.blases[inst.blas_index];
        if (blas.kind == ShapeIR::Kind::TrianglesCulled)
            any_backface_culled = true;
        if (blas.kind == ShapeIR::Kind::Custom)
            any_custom = true;

        for (uint32_t geom_idx = 0; geom_idx < blas.geoms.size(); ++geom_idx) {
            hiprtGeometryBuildInput bi;
            uint32_t base = 0;
            if (!make_build_input(d, blas.geoms[geom_idx], blas.kind, bi,
                                  custom_types, base))
                continue;

            // One entry per EXPANDED instance, which is what hit.instanceID
            // indexes. A shape referenced by several ShapeGroup instances
            // therefore has its records written once per instance rather than
            // shared -- wasteful, and matched by the geometry itself being
            // rebuilt per instance a few lines below. Both are worth fixing
            // together, with a (blas, geometry) -> handle cache, once there is
            // hardware to measure the win on.
            custom_base.push_back(base);

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

    // --- Custom-primitive function table -----------------------------------
    //
    // Built after the scene because it needs the instance-indexed base table,
    // whose length is the expanded instance count. Nothing traverses in
    // between, so the ordering costs nothing.
    if (any_custom) {
        register_isect_source();

        void *base_dev = upload(d, custom_base.data(),
                                custom_base.size() * sizeof(uint32_t));

        HIPRT_CHECK(hiprtCreateFuncTable(context, HIP_ISECT_FN_COUNT,
                                         /* numRayTypes = */ 1,
                                         d->func_table));

        for (uint32_t fn = 0; fn < HIP_ISECT_FN_COUNT; ++fn) {
            const CustomTypeData &td = custom_types[fn];

            // A type with no geometry in this scene still gets an entry: the
            // generated dispatcher indexes the table by geometry type
            // unconditionally, and a null funcDataSets row would be read before
            // anything could establish that no such geometry exists.
            HIPIsectTypeData host{};
            if (!td.records.empty()) {
                host.prims = upload(d, td.records.data(), td.records.size());
                host.base  = (const uint32_t *) base_dev;
            }

            hiprtFuncDataSet set{};
            set.intersectFuncData =
                upload(d, &host, sizeof(HIPIsectTypeData));
            set.filterFuncData = nullptr;
            HIPRT_CHECK(hiprtSetFuncTable(context, d->func_table, fn,
                                          /* rayType = */ 0, set));
        }
    }

    // Bit 0 triangles, bit 1 custom/AABB, bit 2 curves, bit 3 backface culling
    // required -- the same encoding Metal uses. Curves are still unbuildable,
    // so bit 2 is unreachable; the mask is assembled from what was actually
    // built rather than hardcoded, so it stays honest.
    uint32_t types_mask = 0;
    for (const InstanceEntry &inst : sd.instances) {
        ShapeIR::Kind k = sd.blases[inst.blas_index].kind;
        if (is_triangle_kind(k))
            types_mask |= 1u;
        else if (k == ShapeIR::Kind::Custom)
            types_mask |= 1u << 1;
    }
    if (any_backface_culled)
        types_mask |= 1u << 3;

    uint32_t scene_index = jit_hip_configure_scene(
        (void *) d->scene, (void *) d->func_table, geometry_ids_dev,
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
            if (dd->func_table)
                hiprtDestroyFuncTable(dd->context, dd->func_table);
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
