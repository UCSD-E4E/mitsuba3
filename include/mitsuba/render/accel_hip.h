/*
    accel_hip.h -- HIP-RT acceleration backend declarations.

    Modelled on accel_metal.h rather than accel_optix.h, and deliberately so:
    HIP-RT exposes an inline intersector reached from an ordinary compute
    kernel, which is Metal's shape, not OptiX's pipeline/SBT/callable-program
    model. PLAN.md 3.2 has the reasoning; the practical consequence is that
    scene_hip.inl is ~180 lines where scene_optix.inl is 818.
*/

#pragma once

#if defined(MI_ENABLE_HIP)

#include <mitsuba/render/fwd.h>
#include <drjit/array_traverse.h>

NAMESPACE_BEGIN(mitsuba)

/// Opaque handle owning the native HIP-RT objects, see src/render/hip/accel.cpp
struct HIPAccelData;

/// Vectorized GPU ray tracing acceleration via AMD HIP-RT
template <typename Float, typename Spectrum>
struct HIPAccel {
    MI_IMPORT_TYPES(Shape, ShapePtr)
    HIPAccel() = default;
    DRJIT_NON_COPYABLE(HIPAccel)

    ~HIPAccel() { release(); }

    // --- Lifecycle (bodies in scene_hip.inl) ---
    void init(Scene<Float, Spectrum> *scene, const Properties &props);
    void rebuild(Scene<Float, Spectrum> *scene);
    void release();

    static void static_initialization() { }
    static void static_shutdown() { }

    // --- Ray queries (bodies in scene_hip.inl) ---
    PreliminaryIntersection3f ray_intersect_preliminary(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray, Mask coherent,
        bool reorder, UInt32 reorder_hint, uint32_t reorder_hint_bits,
        Mask active) const;
    Mask ray_test(const Scene<Float, Spectrum> *scene, const Ray3f &ray,
                  Mask coherent, Mask active) const;
    /// HIP-RT has no brute-force traversal; defer to the accelerated path.
    SurfaceInteraction3f ray_intersect_naive(
        const Scene<Float, Spectrum> *scene, const Ray3f &ray,
        Mask active) const;

    // --- Declarative traversal (scene handle + recovery tables) ---
    DRJIT_TRAVERSE(HIPAccel, accel_handle, geom_shape_offsets,
                   geom_shape_table)

    /// Opaque handle owning the HIP-RT objects (scene/geometries/buffers)
    HIPAccelData *accel = nullptr;
    /// Dr.Jit scene id from jit_hip_configure_scene(), 0 for empty scenes
    uint32_t scene_index = 0;
    /// Handle variable representing the HIP-RT scene for @dr.freeze
    UInt64 accel_handle;
    /// Per-instance recovery tables resolving \c pi.shape from a hit's
    /// (instance_id, geometry_id), built in scene_hip.inl.
    DynamicBuffer<UInt32> geom_shape_offsets;
    DynamicBuffer<UInt32> geom_shape_table;

private:
    /// Trace \c ray, writing eight result variable indices to \c out. With
    /// \c shadow, an occlusion query writes only ``out[0]``.
    void trace(const Ray3f &ray, Mask active, uint32_t out[8],
               bool shadow) const;
};

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_HIP
