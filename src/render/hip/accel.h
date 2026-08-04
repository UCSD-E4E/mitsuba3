/*
    hip/accel.h -- Plain-C++ interface to the HIP-RT acceleration structure
    builder (src/render/hip/accel.cpp). It consumes the lowered scene (the
    \ref BlasEntry / \ref InstanceEntry descriptors from scene_ir.h), turns it
    into HIP-RT objects, and registers them with Dr.Jit via
    jit_hip_configure_scene().

    Mirrors src/render/metal/accel.h. The one structural difference is
    documented on \ref build_hip_accel.
*/

#pragma once

#if defined(MI_ENABLE_HIP)

#include <mitsuba/core/platform.h>
#include <mitsuba/render/scene_ir.h>
#include <utility>

NAMESPACE_BEGIN(mitsuba)

/// Opaque handle owning the HIP-RT objects of a built scene
struct HIPAccelData;

/**
 * \brief Build the lowered scene's acceleration structures and register them
 * with Dr.Jit.
 *
 * \c compact requests HIP-RT compact BVH builds. Returns the owning
 * \ref HIPAccelData and the scene's JIT variable index (caller-owned), to pass
 * to jit_hip_ray_trace().
 *
 * \remark Unlike Metal and OptiX, a \c hiprtInstance references exactly ONE
 * geometry, and a \c hiprtHit carries no geometry ID. So a \ref BlasEntry
 * holding N same-kind geometries cannot become one instance here. Each
 * (instance, geometry) pair is expanded into its own HIP-RT instance, and the
 * geometry index is supplied back to the kernel through the instance-indexed
 * table handed to jit_hip_configure_scene(). The hit values Mitsuba sees are
 * therefore identical to Metal's, which is what lets scene_hip.inl share
 * scene_metal.inl's recovery logic verbatim.
 */
extern MI_EXPORT_LIB std::pair<HIPAccelData *, uint32_t>
build_hip_accel(const SceneIR &sd, bool compact);

/// Release a scene built with \ref build_hip_accel(): drop the JIT variable
/// reference and free the HIP-RT objects (deferred until no kernel/recording
/// references it).
extern MI_EXPORT_LIB void release_hip_accel(HIPAccelData *accel,
                                            uint32_t scene_index);

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_HIP
