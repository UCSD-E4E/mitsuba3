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

/// Per-primitive data for an ellipsoid (object space: unit sphere).
struct alignas(16) EllipsoidData {
    /// Affine world -> object (unit-sphere) transformation.
    float4 to_object[3];
};

/// PER-SHAPE record for an SDF grid. Unlike every other type here this is one
/// record for the whole shape, and it holds DEVICE POINTERS into arrays that
/// already exist -- the grid and its voxel index table are Dr.Jit arrays, so on
/// HIP they are already HIP memory. Metal has to pack the same information into
/// a self-contained blob because its buffer-binding model forces a copy; HIP
/// does not, and copying a whole SDF grid per scene build would be waste.
/// Mirrors HIPSDFGridData in src/render/hip/shapes.h and OptixSDFGridData.
struct alignas(16) SDFGridData {
    const unsigned int *voxel_indices;
    const float *grid_data;
    unsigned int res_x, res_y, res_z, pad_;
    /// xyz = voxel size (w unused)
    float4 voxel_size;
    /// world -> object affine, row-major, 3 rows
    float4 to_object[3];
};

static_assert(sizeof(SphereData)    == 16, "SphereData layout");
static_assert(sizeof(EllipsoidData) == 48, "EllipsoidData layout");
static_assert(sizeof(DiskData)     == 48, "DiskData layout");
static_assert(sizeof(CylinderData) == 64, "CylinderData layout");
// The only record here NOT mirrored from <mitsuba/render/shapedata.h>, and the
// only one holding pointers -- so it is the one whose host/device agreement
// nothing else checks. Pinned on both sides; see HIPSDFGridData in
// src/render/hip/shapes.h.
static_assert(sizeof(SDFGridData)  == 96, "SDFGridData layout");

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
    /// Indexed by hit.instanceID: BYTE offset of that geometry's slice.
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

/// Base of this geometry's slice. `base` is a BYTE offset so that fixed-size
/// per-primitive layouts and variable-length per-shape ones share one table.
__device__ inline const unsigned char *mi_slice(const void *data,
                                                const hiprtHit &hit) {
    const MiIsectTypeData *d = (const MiIsectTypeData *) data;
    return (const unsigned char *) d->prims + d->base[hit.instanceID];
}

/// This geometry's record for the primitive being tested -- for shapes that
/// describe one record PER PRIMITIVE (sphere, disk, cylinder, ellipsoid).
template <typename T>
__device__ inline const T *mi_prim(const void *data, const hiprtHit &hit) {
    return (const T *) mi_slice(data, hit) + hit.primID;
}

/// This geometry's single record -- for shapes that describe one record PER
/// SHAPE (SDFGrid), where hit.primID indexes inside the record instead.
template <typename T>
__device__ inline const T *mi_shape(const void *data, const hiprtHit &hit) {
    return (const T *) mi_slice(data, hit);
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

// ---------------------------------------------------------------------------
//  Ellipsoids (one ellipsoid per primitive; object space is the unit sphere)
// ---------------------------------------------------------------------------

/*
    Transcribed from intersection_ellipsoids() in
    src/render/metal/intersection_functions.metal:229, itself following
    src/shapes/optix/ellipsoids.cuh.

    Note the differences from mi_hip_isect_sphere, all of which are deliberate
    and come from OptiX's semantics rather than from the geometry:

      * the ray is mapped into unit-sphere space by an affine, so the quadratic
        is against a unit sphere rather than a (centre, radius) pair;
      * `backfacing` -- a ray originating INSIDE the ellipsoid is rejected
        outright, which the sphere intersector does not do;
      * only `near_t` is ever accepted, never `far_t`.

    Keeping these visible is the point of transcribing rather than sharing code
    with the sphere: the two look similar and are not.
*/
__device__ bool mi_hip_isect_ellipsoids(const hiprtRay &ray, const void *data,
                                        void *payload, hiprtHit &hit) {
    (void) payload;
    const mi_shapedata::EllipsoidData *e =
        mi_prim<mi_shapedata::EllipsoidData>(data, hit);

    float3 ro = mi_affine_point(e->to_object, ray.origin);
    float3 rd = mi_affine_vector(e->to_object, ray.direction);

    // Perpendicular-plane projection, as in the sphere: shift the origin to the
    // foot of the perpendicular from the centre onto the ray so the
    // discriminant is well conditioned for distant or grazing rays.
    float plane_t = -mi_dot(ro, rd) / sqrtf(mi_dot(rd, rd));
    float3 plane_p = mi_fma(rd, plane_t, ro);

    if (plane_t == 0.0f && sqrtf(mi_dot(plane_p, plane_p)) > 1.0f)
        return false;

    float A = mi_dot(rd, rd);
    float B = 2.0f * mi_dot(plane_p, rd);
    float C = mi_dot(plane_p, plane_p) - 1.0f;

    float near_t, far_t;
    bool ok = mi_solve_quadratic(A, B, C, near_t, far_t);

    near_t += plane_t;
    far_t  += plane_t;

    bool out_bounds = !(near_t <= ray.maxT && far_t >= 0.0f);
    bool in_bounds  = near_t < 0.0f && far_t > ray.maxT;
    bool backfacing = near_t < 0.0f;

    if (!ok || out_bounds || in_bounds || backfacing ||
        near_t < ray.minT || near_t > ray.maxT)
        return false;

    hit.t  = near_t;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}

// ---------------------------------------------------------------------------
//  SDF grid (one AABB per filled voxel)
// ---------------------------------------------------------------------------

/// Slab test against a voxel's object-space AABB.
__device__ inline bool mi_sdf_aabb(const float3 &ro, const float3 &rd,
                                   const float3 &bbmin, const float3 &bbmax,
                                   float &t_min, float &t_max) {
    float3 inv_d = make_float3(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);
    float t1x = (bbmin.x - ro.x) * inv_d.x, t2x = (bbmax.x - ro.x) * inv_d.x;
    float t1y = (bbmin.y - ro.y) * inv_d.y, t2y = (bbmax.y - ro.y) * inv_d.y;
    float t1z = (bbmin.z - ro.z) * inv_d.z, t2z = (bbmax.z - ro.z) * inv_d.z;

    t_min = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
    t_max = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));
    return t_max >= fmaxf(t_min, 0.0f);
}

/*
    Transcribed from intersection_sdfgrid() in
    src/render/metal/intersection_functions.metal:323, which follows the OptiX
    __intersection__sdfgrid and Hansson-Soderlund et al. 2022.

    The SDF restricted to a ray within one voxel is a cubic in t (the trilinear
    interpolant composed with a line). The coefficients c0..c3 are that cubic;
    the root is found by bracketing at the derivative's roots and then a
    regula-falsi / bisection hybrid.

    hit.primID selects the VOXEL, not a record: this is a per-shape record, so
    mi_shape() is used and primID indexes voxel_indices inside it.
*/
__device__ bool mi_hip_isect_sdfgrid(const hiprtRay &ray, const void *data,
                                     void *payload, hiprtHit &hit) {
    (void) payload;
    const mi_shapedata::SDFGridData *h =
        mi_shape<mi_shapedata::SDFGridData>(data, hit);

    unsigned int vi = h->voxel_indices[hit.primID];
    unsigned int x_len = h->res_x - 1u;
    unsigned int y_len = h->res_y - 1u;

    // Voxel index encoding: vi = vx + vy * x_len + vz * x_len * y_len
    // (SDFGrid::build_bboxes()).
    unsigned int vx = vi % x_len;
    unsigned int vy = ((vi - vx) / x_len) % y_len;
    unsigned int vz = (vi - vx - vy * x_len) / (x_len * y_len);

    float3 ro = mi_affine_point(h->to_object, ray.origin);
    float3 rd = mi_affine_vector(h->to_object, ray.direction);

    float3 vox   = make_float3((float) vx, (float) vy, (float) vz);
    float3 vsize = make_float3(h->voxel_size.x, h->voxel_size.y,
                               h->voxel_size.z);
    float3 bbmin = make_float3(vox.x * vsize.x, vox.y * vsize.y,
                               vox.z * vsize.z);
    float3 bbmax = make_float3(bbmin.x + vsize.x, bbmin.y + vsize.y,
                               bbmin.z + vsize.z);

    float t_bbox_beg = 0.0f, t_bbox_end = 0.0f;
    if (!mi_sdf_aabb(ro, rd, bbmin, bbmax, t_bbox_beg, t_bbox_end))
        return false;

    t_bbox_beg = fmaxf(t_bbox_beg, 0.0f);
    if (t_bbox_end < t_bbox_beg)
        return false;

    // Ray in voxel-local [0,1]^3: p_voxel = p_object / voxel_size - voxel_pos.
    float3 inv_vs = make_float3(1.0f / vsize.x, 1.0f / vsize.y, 1.0f / vsize.z);
    float3 ro_v = make_float3(ro.x * inv_vs.x - vox.x, ro.y * inv_vs.y - vox.y,
                              ro.z * inv_vs.z - vox.z);
    float3 rd_v = make_float3(rd.x * inv_vs.x, rd.y * inv_vs.y,
                              rd.z * inv_vs.z);

    unsigned int sx = h->res_x, sy = h->res_y;
    const float *grid = h->grid_data;
#define MI_SDF_AT(X, Y, Z) grid[(Z) * sy * sx + (Y) * sx + (X)]
    float s000 = MI_SDF_AT(vx,      vy,      vz     );
    float s100 = MI_SDF_AT(vx + 1u, vy,      vz     );
    float s010 = MI_SDF_AT(vx,      vy + 1u, vz     );
    float s110 = MI_SDF_AT(vx + 1u, vy + 1u, vz     );
    float s001 = MI_SDF_AT(vx,      vy,      vz + 1u);
    float s101 = MI_SDF_AT(vx + 1u, vy,      vz + 1u);
    float s011 = MI_SDF_AT(vx,      vy + 1u, vz + 1u);
    float s111 = MI_SDF_AT(vx + 1u, vy + 1u, vz + 1u);
#undef MI_SDF_AT

    float3 p0 = mi_fma(rd_v, t_bbox_beg, ro_v);
    float o_x = p0.x, o_y = p0.y, o_z = p0.z;
    float d_x = rd_v.x, d_y = rd_v.y, d_z = rd_v.z;

    float a  = s101 - s001;
    float k0 = s000;
    float k1 = s100 - s000;
    float k2 = s010 - s000;
    float k3 = s110 - s010 - k1;
    float k4 = k0 - s001;
    float k5 = k1 - a;
    float k6 = k2 - (s011 - s001);
    float k7 = k3 - (s111 - s011 - a);
    float m0 = o_x * o_y;
    float m1 = d_x * d_y;
    float m2 = fmaf(o_x, d_y, o_y * d_x);
    float m3 = fmaf(k5, o_z, -k1);
    float m4 = fmaf(k6, o_z, -k2);
    float m5 = fmaf(k7, o_z, -k3);

    float c0 = fmaf(k4, o_z, -k0) + fmaf(o_x, m3, fmaf(o_y, m4, m0 * m5));
    float c1 = fmaf(d_x, m3, d_y * m4) + m2 * m5 +
               d_z * (k4 + fmaf(k5, o_x, fmaf(k6, o_y, k7 * m0)));
    float c2 = fmaf(m1, m5, d_z * (fmaf(k5, d_x, fmaf(k6, d_y, k7 * m2))));
    float c3 = k7 * m1 * d_z;

    float t_beg = 0.0f;
    float t_end = t_bbox_end - t_bbox_beg;

    // The cubic, Horner form. A macro rather than a lambda so it is visibly the
    // same expression the MSL and OptiX versions evaluate.
#define MI_SDF_EVAL(t) fmaf(fmaf(fmaf(c3, (t), c2), (t), c1), (t), c0)

    float t_hit = 0.0f;
    bool found = false;

    if (c3 != 0.0f) {
        // Cubic: bracket at the derivative's roots, then solve numerically.
        float root_0 = 0.0f, root_1 = 0.0f;
        bool has_drv = mi_solve_quadratic(c3 * 3.0f, c2 * 2.0f, c1,
                                          root_0, root_1);
        float t_near = t_beg, t_far = t_end;
        float f_near = MI_SDF_EVAL(t_near);
        float f_far  = MI_SDF_EVAL(t_far);

        if (has_drv) {
            if (t_near <= root_0 && root_0 <= t_far) {
                float fr = MI_SDF_EVAL(root_0);
                if (f_near * fr <= 0.0f) { t_far = root_0; f_far = fr; }
                else                     { t_near = root_0; f_near = fr; }
            }
            if (t_near <= root_1 && root_1 <= t_far) {
                float fr = MI_SDF_EVAL(root_1);
                if (f_near * fr <= 0.0f) { t_far = root_1; f_far = fr; }
                else                     { t_near = root_1; f_near = fr; }
            }
        }
        if (f_near * f_far > 0.0f)
            return false;

        // Regula falsi with bracket maintenance, as in the MSL and OptiX
        // versions -- 50 iterations or a 1e-5 bracket, whichever comes first.
        float t = 0.0f;
        for (int i = 0; i < 50; ++i) {
            t = t_near + (t_far - t_near) * (-f_near / (f_far - f_near));
            float f_t = MI_SDF_EVAL(t);
            if (f_t * f_near <= 0.0f) { t_far = t;  f_far = f_t; }
            else                      { t_near = t; f_near = f_t; }
            if (fabsf(t_far - t_near) < 1e-5f)
                break;
        }
        if (t_beg <= t && t <= t_end) {
            t_hit = t_bbox_beg + t;
            found = true;
        }
    } else {
        // Degenerate cubic: quadratic or linear.
        float r0 = 0.0f, r1 = 0.0f;
        if (mi_solve_quadratic(c2, c1, c0, r0, r1)) {
            if (t_beg <= r0 && r0 <= t_end) {
                t_hit = t_bbox_beg + r0;
                found = true;
            } else if (t_beg <= r1 && r1 <= t_end) {
                t_hit = t_bbox_beg + r1;
                found = true;
            }
        }
    }

#undef MI_SDF_EVAL

    if (!found || t_hit < ray.minT || t_hit >= ray.maxT)
        return false;

    hit.t  = t_hit;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}
