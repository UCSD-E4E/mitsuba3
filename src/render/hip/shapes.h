/*
    hip/shapes.h -- HIP-RT geometry types for custom (implicit) shapes.
    The per-primitive POD layouts live in <mitsuba/render/shapedata.h>, and the
    device code that reads them in hip/intersection_functions.hip.
*/

#pragma once

#if defined(MI_ENABLE_HIP)

#include <mitsuba/core/platform.h>
#include <mitsuba/render/shapedata.h>
#include <mitsuba/render/fwd.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

NAMESPACE_BEGIN(mitsuba)

/// HIP-RT geometry type indices for custom shapes.
///
/// A geometry type selects a row of the hiprtFuncTable, which is what
/// hiprtBuildTraceKernels() turns into a `switch` in its generated
/// `intersectFunc`. The index is stored in hiprtGeometryBuildInput::geomType.
///
/// The order must match \ref hip_isect_fn_names, because HIP-RT pairs the two
/// positionally through its funcNameSets argument.
///
/// This enum is deliberately SHORTER than Metal's MetalIntersectionFn: it lists
/// what actually has a device implementation, not what Mitsuba has shapes for.
/// Every entry here must have a definition in
/// src/render/hip/intersection_functions.hip -- HIP-RT forward-declares each
/// name it is given, so a missing definition is a link failure at render time
/// rather than a build failure here. Shapes with no entry are refused by
/// build_hip_accel() with a message naming them.
enum HIPIntersectionFn : uint32_t {
    HIP_ISECT_FN_SPHERE   = 0,
    HIP_ISECT_FN_DISK     = 1,
    HIP_ISECT_FN_CYLINDER = 2,
    HIP_ISECT_FN_COUNT    = 3
};

/// Device function names, in \ref HIPIntersectionFn order.
static const char *const hip_isect_fn_names[] = {
    "mi_hip_isect_sphere",
    "mi_hip_isect_disk",
    "mi_hip_isect_cylinder"
};

static_assert(std::size(hip_isect_fn_names) == HIP_ISECT_FN_COUNT,
              "hip_isect_fn_names must have one entry per HIPIntersectionFn.");

/// Map a custom shape's \ref ShapeType to its HIP-RT geometry type. Returns
/// \c HIP_ISECT_FN_COUNT for a shape this backend cannot intersect yet.
///
/// This function is the ONE place that decides what is supported. Anything that
/// wants to describe the answer in prose must ask \ref hip_supported_shapes()
/// rather than restate it -- a hand-written "only spheres are implemented"
/// went stale within an hour of being written.
inline uint32_t hip_fn_index(ShapeType type) {
    switch (type) {
        case ShapeType::Sphere:   return HIP_ISECT_FN_SPHERE;
        case ShapeType::Disk:     return HIP_ISECT_FN_DISK;
        case ShapeType::Cylinder: return HIP_ISECT_FN_CYLINDER;
        default:                  return HIP_ISECT_FN_COUNT;
    }
}

/// Human-readable list of the custom shapes \ref hip_fn_index accepts, derived
/// from it rather than maintained beside it.
inline std::string hip_supported_shapes() {
    static const std::pair<ShapeType, const char *> all[] = {
        { ShapeType::Sphere,     "sphere"     },
        { ShapeType::Disk,       "disk"       },
        { ShapeType::Cylinder,   "cylinder"   },
        { ShapeType::Ellipsoids, "ellipsoids" },
        { ShapeType::SDFGrid,    "sdfgrid"    },
    };
    std::string yes, no;
    for (auto &e : all) {
        std::string &dst = hip_fn_index(e.first) < HIP_ISECT_FN_COUNT ? yes : no;
        if (!dst.empty())
            dst += ", ";
        dst += e.second;
    }
    return "supported: " + (yes.empty() ? std::string("none") : yes) +
           "; not yet: " + (no.empty() ? std::string("none") : no);
}

/// The per-geometry-type struct handed to a HIP intersection function as its
/// `data` argument. Must match MiIsectTypeData in
/// src/render/hip/intersection_functions.hip.
///
/// HIP-RT's `data` is per geometry TYPE, not per geometry, so it cannot point
/// straight at one shape's records; `base` closes that gap. Indexing it by
/// instance is exact rather than approximate -- build_hip_accel() expands every
/// (instance, geometry) pair into its own HIP-RT instance, so an instance
/// identifies a geometry uniquely.
struct HIPIsectTypeData {
    /// Device pointer to the combined array of this type's records.
    const void *prims;
    /// Device pointer to uint32_t[n_instances]: first record of that geometry.
    const uint32_t *base;
};

NAMESPACE_END(mitsuba)

#endif // MI_ENABLE_HIP
