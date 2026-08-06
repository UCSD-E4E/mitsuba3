/*
    scene_hip.inl -- Templated half of the HIP-RT backend: lowers the scene for
    the acceleration structure builder (src/render/hip/accel.cpp) and dispatches
    rays through jit_hip_ray_trace(). State and declarations live in
    include/mitsuba/render/accel_hip.h.

    Near-identical to scene_metal.inl by construction: drjit-core/hip.h was
    written against the same eight-output hit record as drjit-core/metal.h
    precisely so this layer could be shared (PLAN.md 3.2). Where the two differ
    is noted inline.
*/

#include <drjit-core/hip.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/scene_ir.h>
#include "hip/accel.h"

NAMESPACE_BEGIN(mitsuba)

/// Build the raw recovery tables that resolve \c pi.shape from a HIP-RT hit.
///
/// Same two-level scheme as Metal -- offsets[instance] + geometry gives an
/// index into a flat shape-id table -- even though HIP-RT reaches it
/// differently. A hiprtHit reports only (instanceID, primID): it has no
/// geometry ID, because a hiprtInstance references exactly one geometry, so
/// "which geometry" is a property of the instance. build_hip_accel() therefore
/// expands each (instance, geometry) pair into its own HIP-RT instance and
/// supplies the geometry index through the instance-indexed table passed to
/// jit_hip_configure_scene(). By the time the values reach this code they mean
/// what Metal's mean, which is why the lookup below is unchanged.
///
/// THE INDEX IS THE EXPANDED INSTANCE ID. That is the whole subtlety here.
/// \c hit.instanceID counts HIP-RT instances, of which build_hip_accel() makes
/// one per (SceneIR instance, geometry) pair -- not one per SceneIR instance.
/// So \c offsets needs one entry per expanded instance, and every expanded
/// instance drawn from the same BLAS repeats that BLAS's table base; the
/// geometry index that distinguishes them arrives separately, as output 6.
///
/// Indexing this by SceneIR instance instead reads past the end of the buffer
/// for every geometry after the first in a BLAS. It is a quiet failure: SceneIR
/// buckets same-kind geometry into ONE BLAS, so a scene with a single mesh (or
/// with only one BLAS, where every base is 0 and reading zeros off the end
/// happens to give the right answer) looks perfectly correct, and a scene with
/// two meshes silently shades each hit with its neighbour's BSDF.
///
/// The expansion order below must match build_hip_accel()'s exactly -- both
/// walk sd.instances, then geoms in order.
static void build_recovery_table_data(const SceneIR &sd,
                                      std::vector<uint32_t> &offsets,
                                      std::vector<uint32_t> &table) {
    offsets.clear();
    table.clear();
    for (const InstanceEntry &inst : sd.instances) {
        const std::vector<ShapeIR> &geoms = sd.blases[inst.blas_index].geoms;
        uint32_t base = (uint32_t) table.size();
        for (size_t i = 0; i < geoms.size(); ++i)
            offsets.push_back(base);
        for (const ShapeIR &g : geoms)
            table.push_back(jit_registry_id(g.ctx));
    }
}

// -----------------------------------------------------------------------
//  HIPAccel<Float, Spectrum> -- lifecycle
// -----------------------------------------------------------------------

template <typename Float, typename Spectrum>
void HIPAccel<Float, Spectrum>::init(Scene<Float, Spectrum> *scene,
                                     const Properties & /*props*/) {
    SceneIR sd = SceneIRBuilder<Float, Spectrum>::build(scene);

    if (sd.instances.empty()) {
        Log(Debug, "accel_init_hip(): scene contains no shapes.");
        return;
    }

    std::vector<uint32_t> offsets, table;
    build_recovery_table_data(sd, offsets, table);
    using UInt32 = dr::uint32_array_t<Float>;
    geom_shape_offsets =
        dr::load<DynamicBuffer<UInt32>>(offsets.data(), offsets.size());
    geom_shape_table =
        dr::load<DynamicBuffer<UInt32>>(table.data(), table.size());

    std::tie(accel, scene_index) =
        build_hip_accel(sd, scene->m_compact_accel);

    accel_handle = UInt64::steal(jit_hip_scene_owner_handle(scene_index));
}

template <typename Float, typename Spectrum>
void HIPAccel<Float, Spectrum>::rebuild(Scene<Float, Spectrum> *scene) {
    // Build-and-swap: a geometry edit drops the previous scene and registers a
    // fresh one. Frozen functions follow the traversed handle on the next
    // replay, so no in-place scene mutation is needed.
    release();
    Properties props;
    init(scene, props);
}

template <typename Float, typename Spectrum>
void HIPAccel<Float, Spectrum>::release() {
    if (!accel && scene_index == 0)
        return; // Empty scene or already released
    accel_handle = 0;
    release_hip_accel(accel, scene_index);
    accel = nullptr;
    scene_index = 0;
}

// -----------------------------------------------------------------------
//  HIPAccel<Float, Spectrum> -- ray queries
// -----------------------------------------------------------------------

template <typename Float, typename Spectrum>
void HIPAccel<Float, Spectrum>::trace(const Ray3f &ray, Mask active,
                                      uint32_t out[8], bool shadow) const {
    using Single = dr::float32_array_t<Float>;
    dr::Array<Single, 3> ray_o(ray.o), ray_d(ray.d);
    Single ray_tmin(0.f), ray_tmax(ray.maxt);

    // Be careful with 'ray.maxt' in double precision variants. Unlike Metal,
    // HIP has real double variants (gfx90a has full-rate FP64), so this clamp
    // is reachable here rather than dead code.
    if constexpr (!std::is_same_v<Single, Float>)
        ray_tmax = dr::minimum(ray_tmax, dr::Largest<Single>);

    uint32_t args[8] = {
        ray_o.x().index(), ray_o.y().index(), ray_o.z().index(),
        ray_d.x().index(), ray_d.y().index(), ray_d.z().index(),
        ray_tmin.index(), ray_tmax.index()
    };

    jit_hip_ray_trace(8, args, active.index(), out, 8, scene_index, shadow);
}

template <typename Float, typename Spectrum>
typename HIPAccel<Float, Spectrum>::PreliminaryIntersection3f
HIPAccel<Float, Spectrum>::ray_intersect_preliminary(
    const Scene<Float, Spectrum> * /*scene*/, const Ray3f &ray, Mask /*coh*/,
    bool /*reorder*/, UInt32 /*reorder_hint*/, uint32_t /*reorder_hint_bits*/,
    Mask active) const {
    using Single = dr::float32_array_t<Float>;

    PreliminaryIntersection3f pi = dr::zeros<PreliminaryIntersection3f>();
    if (scene_index == 0) // Empty scene: every ray misses
        return pi;

    // out: [valid, distance, bary_u, bary_v, instance_id, primitive_id,
    //       geometry_id, user_instance_id]
    uint32_t out[8];
    trace(ray, active, out, /* shadow = */ false);

    Mask valid = Mask::steal(out[0]);

    pi.valid      = valid;
    pi.t          = Float(Single::steal(out[1]));
    pi.prim_uv    = Point2f(Float(Single::steal(out[2])),
                            Float(Single::steal(out[3])));
    pi.prim_index = UInt32::steal(out[5]);

    UInt32 instance_id = UInt32::steal(out[4]);
    UInt32 geometry_id = UInt32::steal(out[6]);

    // The hit shape's registry id is the (instance, geometry) entry of the
    // recovery table, so pi.shape names the actual child for every hit. The
    // gathers are masked, so missed lanes read 0 (a null shape).
    UInt32 off      = dr::gather<UInt32>(geom_shape_offsets, instance_id, valid);
    UInt32 shape_id = dr::gather<UInt32>(geom_shape_table, off + geometry_id,
                                         valid);
    pi.shape = dr::reinterpret_array<ShapePtr, UInt32>(shape_id);

    // userID is the owning Instance's registry id, or 0 (a null shape) for a
    // top-level hit. Missed lanes read 0 too. Consecutive TLAS instances of one
    // ShapeGroup share a userID, which is correct.
    UInt32 user_id = dr::select(valid, UInt32::steal(out[7]), dr::zeros<UInt32>());
    pi.instance = dr::reinterpret_array<ShapePtr, UInt32>(user_id);

    return pi;
}

template <typename Float, typename Spectrum>
typename HIPAccel<Float, Spectrum>::Mask
HIPAccel<Float, Spectrum>::ray_test(const Scene<Float, Spectrum> * /*scene*/,
                                    const Ray3f &ray, Mask /*coherent*/,
                                    Mask active) const {
    if (scene_index == 0) // Empty scene: no occluders
        return dr::zeros<Mask>(dr::width(ray.o));

    uint32_t out[8];
    trace(ray, active, out, /* shadow = */ true);

    // A shadow trace computes only out[0] (the hit flag). out[1..7] are left
    // untouched (see jit_hip_ray_trace), so steal just the hit flag.
    return Mask::steal(out[0]);
}

template <typename Float, typename Spectrum>
typename HIPAccel<Float, Spectrum>::SurfaceInteraction3f
HIPAccel<Float, Spectrum>::ray_intersect_naive(
    const Scene<Float, Spectrum> *scene, const Ray3f &ray, Mask active) const {
    // HIP-RT has no brute-force path; route through the accelerated query.
    return scene->ray_intersect(ray, active);
}

NAMESPACE_END(mitsuba)
