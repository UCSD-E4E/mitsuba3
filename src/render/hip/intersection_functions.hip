/*
    intersection_functions.hip -- HIP C++ source for the custom intersection
    functions used by HIP-RT ray tracing.

    THIS FILE IS NOT COMPILED BY THE BUILD. Its text is embedded verbatim into
    libmitsuba-render (see src/render/CMakeLists.txt) and handed to
    jit_hip_set_isect_source(), which prepends it to every traversing kernel
    drjit-core generates. That indirection is forced: hiprtBuildTraceKernels()
    GENERATES the intersectFunc/filterFunc dispatchers from its funcNameSets
    argument and compiles them together with the kernel, so the definitions
    cannot be precompiled the way Metal ships a .metallib
    (BACKEND_NOTES 7a-3, 11o).

    Consequences worth keeping in mind while editing:

      * It is compiled by hiprtc (or NVRTC under the CUDA shim) with only
        HIP-RT's own include directory on the search path. Mitsuba headers are
        NOT available -- hence the POD layouts are restated below rather than
        included from <mitsuba/render/shapedata.h>, with static_asserts pinning
        them to the same sizes.

      * A syntax error here surfaces as a failure to build a *render kernel*,
        possibly far from any change to this file. tools/hip_validate/isect_probe
        compiles it standalone in a fraction of the time; use it.

    The math is transcribed from src/render/metal/intersection_functions.metal,
    which is itself a transcription of the OptiX .cuh headers. Keeping all three
    line-comparable is deliberate.
*/

#include <hiprt/hiprt_device.h>

// ---------------------------------------------------------------------------
//  Per-primitive data layouts -- must match <mitsuba/render/shapedata.h>
// ---------------------------------------------------------------------------

namespace mi_shapedata {

/// Per-primitive data for a sphere (world space). shapedata::SphereData.
struct alignas(16) SphereData {
    /// xyz = center, w = radius
    float4 center_radius;
};

/// Per-primitive data for a disk (object space: z=0 plane, unit radius).
struct alignas(16) DiskData {
    /// Affine world -> object transformation, three rows.
    float4 to_object[3];
};

/// Per-primitive data for a cylinder (object space: z-axis, [0,length], radius).
struct alignas(16) CylinderData {
    float4 to_object[3];
    /// x = length, y = radius (z, w unused).
    float4 params;
};

static_assert(sizeof(SphereData)   == 16, "SphereData layout");
static_assert(sizeof(DiskData)     == 48, "DiskData layout");
static_assert(sizeof(CylinderData) == 64, "CylinderData layout");

} // namespace mi_shapedata

// ---------------------------------------------------------------------------
//  Locating a primitive's data
// ---------------------------------------------------------------------------

/*
    HIP-RT hands the intersection function one `data` pointer per GEOMETRY TYPE
    (it is funcDataSets[numGeomTypes * rayType + geomType]), not per geometry.
    So `data` points at the struct below: the combined per-type primitive array
    plus a table giving each instance's first element in it.

    Indexing by instance works because build_hip_accel() expands every
    (instance, geometry) pair into its own HIP-RT instance -- a hiprtInstance
    references exactly one geometry (src/hip_scene.h). Metal needs a two-level
    (instance -> BLAS record -> geometry) lookup precisely because its
    acceleration structures nest several geometries inside one instance.

    `hit.instanceID` and `hit.primID` are both populated before the call
    (hiprt_device_impl.h:1032 and :1043).
*/
struct MiIsectTypeData {
    /// Combined array of this type's per-primitive records.
    const void *prims;
    /// Indexed by hit.instanceID: index of that geometry's first record.
    const unsigned int *base;
};

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

/// Numerically stable quadratic solver, ordered roots x0 <= x1.
/// Mirrors optix/math.cuh::solve_quadratic and the MSL copy.
__device__ inline bool mi_solve_quadratic(float a, float b, float c,
                                          float &x0, float &x1) {
    bool linear_case = (a == 0.0f);
    if (linear_case && b == 0.0f)
        return false;

    // Linear-case root (b != 0 here); overwritten below in the quadratic case.
    x0 = x1 = -c / b;
    if (linear_case)
        return true;

    float discrim = fmaf(b, b, -4.0f * a * c);
    if (discrim < 0.0f)
        return false;

    float temp = -0.5f * (b + copysignf(sqrtf(discrim), b));
    float x0p = temp / a;
    float x1p = c / temp;
    x0 = fminf(x0p, x1p);
    x1 = fmaxf(x0p, x1p);
    return true;
}

__device__ inline float mi_dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 mi_sub(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 mi_fma(const float3 &a, float t, const float3 &b) {
    return make_float3(fmaf(a.x, t, b.x), fmaf(a.y, t, b.y), fmaf(a.z, t, b.z));
}

/// Apply a row-major affine (3 rows, implicit (0,0,0,1)) to a point.
__device__ inline float3 mi_affine_point(const float4 *m, const float3 &p) {
    return make_float3(m[0].x * p.x + m[0].y * p.y + m[0].z * p.z + m[0].w,
                       m[1].x * p.x + m[1].y * p.y + m[1].z * p.z + m[1].w,
                       m[2].x * p.x + m[2].y * p.y + m[2].z * p.z + m[2].w);
}

/// Apply a row-major affine (3 rows) to a vector -- no translation.
__device__ inline float3 mi_affine_vector(const float4 *m, const float3 &v) {
    return make_float3(m[0].x * v.x + m[0].y * v.y + m[0].z * v.z,
                       m[1].x * v.x + m[1].y * v.y + m[1].z * v.z,
                       m[2].x * v.x + m[2].y * v.y + m[2].z * v.z);
}

/// This geometry's record for the primitive being tested.
template <typename T>
__device__ inline const T *mi_prim(const void *data, const hiprtHit &hit) {
    const MiIsectTypeData *d = (const MiIsectTypeData *) data;
    return (const T *) d->prims + d->base[hit.instanceID] + hit.primID;
}

// ---------------------------------------------------------------------------
//  Sphere
// ---------------------------------------------------------------------------

/*
    Signature fixed by HIP-RT's generated dispatcher; see
    tools/hip_validate/isect_probe/README.md in drjit-core, which recovers it
    from the shipped library AND verifies it by building both a correct and a
    deliberately wrong version.

    `ray` arrives in OBJECT space -- transformRay() has already applied the
    instance transform. Writing hit.t/uv and returning true accepts the hit;
    HIP-RT then clamps against the current closest hit itself.

    hit.normal is left alone: Mitsuba recomputes the normal analytically from
    (shape, prim_index, uv) in Sphere::compute_surface_interaction(), so a
    normal written here would be recomputed and discarded. The same is true on
    Metal and OptiX.
*/
__device__ bool mi_hip_isect_sphere(const hiprtRay &ray, const void *data,
                                    void *payload, hiprtHit &hit) {
    (void) payload;
    const mi_shapedata::SphereData *s =
        mi_prim<mi_shapedata::SphereData>(data, hit);

    float3 center = make_float3(s->center_radius.x, s->center_radius.y,
                                s->center_radius.z);
    float  radius = s->center_radius.w;

    float3 d = ray.direction;
    float3 l = mi_sub(ray.origin, center);

    // Recentre the quadratic on the point of closest approach rather than the
    // ray origin (optix/sphere.cuh): for a distant origin the b^2 - 4ac of the
    // original quadratic is a difference of two large nearly-equal numbers, and
    // the hit is lost to cancellation.
    float plane_t = -mi_dot(l, d) / sqrtf(mi_dot(d, d));
    float3 plane_p = mi_fma(d, plane_t, ray.origin);
    float3 o = mi_sub(plane_p, center);

    if (plane_t == 0.0f && sqrtf(mi_dot(o, o)) > radius)
        return false;

    float A = mi_dot(d, d);
    float B = 2.0f * mi_dot(o, d);
    float C = mi_dot(o, o) - radius * radius;

    float near_t, far_t;
    bool ok = mi_solve_quadratic(A, B, C, near_t, far_t);

    near_t += plane_t;
    far_t  += plane_t;

    bool out_bounds = !(near_t <= ray.maxT && far_t >= 0.0f);
    bool in_bounds  = near_t < ray.minT && far_t > ray.maxT;

    float t = (near_t < 0.0f ? far_t : near_t);

    if (!ok || out_bounds || in_bounds || t < ray.minT || t > ray.maxT)
        return false;

    hit.t  = t;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}

// ---------------------------------------------------------------------------
//  Disk (object space: z = 0 plane, unit radius)
// ---------------------------------------------------------------------------

__device__ bool mi_hip_isect_disk(const hiprtRay &ray, const void *data,
                                  void *payload, hiprtHit &hit) {
    (void) payload;
    const mi_shapedata::DiskData *d = mi_prim<mi_shapedata::DiskData>(data, hit);

    float3 ro = mi_affine_point(d->to_object, ray.origin);
    float3 rd = mi_affine_vector(d->to_object, ray.direction);

    float t = -ro.z / rd.z;
    float3 local = mi_fma(rd, t, ro);

    if (!(local.x * local.x + local.y * local.y <= 1.0f) ||
        t < ray.minT || t > ray.maxT)
        return false;

    hit.t  = t;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}

// ---------------------------------------------------------------------------
//  Cylinder (object space: z axis, z in [0, length], given radius)
// ---------------------------------------------------------------------------

__device__ bool mi_hip_isect_cylinder(const hiprtRay &ray, const void *data,
                                      void *payload, hiprtHit &hit) {
    (void) payload;
    const mi_shapedata::CylinderData *c =
        mi_prim<mi_shapedata::CylinderData>(data, hit);
    float length = c->params.x, radius = c->params.y;

    float3 ro = mi_affine_point(c->to_object, ray.origin);
    float3 rd = mi_affine_vector(c->to_object, ray.direction);

    // The quadratic is in the xy projection only -- an infinite cylinder --
    // and the z extent is applied afterwards by choosing which root to take.
    float A = rd.x * rd.x + rd.y * rd.y;
    float B = 2.0f * (rd.x * ro.x + rd.y * ro.y);
    float C = ro.x * ro.x + ro.y * ro.y - radius * radius;

    float near_t, far_t;
    bool ok = mi_solve_quadratic(A, B, C, near_t, far_t);

    bool out_bounds = !(near_t <= ray.maxT && far_t >= ray.minT);

    float z_pos_near = ro.z + rd.z * near_t;
    float z_pos_far  = ro.z + rd.z * far_t;

    bool in_bounds = near_t < ray.minT && far_t > ray.maxT;

    bool valid =
        ok && !out_bounds && !in_bounds &&
        ((z_pos_near >= 0.0f && z_pos_near <= length && near_t > ray.minT) ||
         (z_pos_far  >= 0.0f && z_pos_far  <= length && far_t  < ray.maxT));

    float t = (z_pos_near >= 0.0f && z_pos_near <= length && near_t >= 0.0f)
              ? near_t : far_t;

    if (!valid || t < ray.minT || t > ray.maxT)
        return false;

    hit.t  = t;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}
