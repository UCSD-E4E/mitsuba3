#pragma once

#include <drjit/fwd.h>
#include <mitsuba/mitsuba.h>
#include <type_traits>

NAMESPACE_BEGIN(mitsuba)

// =============================================================
//! @{ \name Color mode traits
// =============================================================

NAMESPACE_BEGIN(detail)

template <typename Spectrum> struct spectrum_traits { };

template <typename Float>
struct spectrum_traits<Color<Float, 1>> {
    using Scalar                             = Color<dr::scalar_t<Float>, 1>;
    using Wavelength                         = Color<Float, 0>;
    using Unpolarized                        = Color<Float, 1>;
    static constexpr bool is_monochromatic   = true;
    static constexpr bool is_rgb             = false;
    static constexpr bool is_spectral        = false;
    static constexpr bool is_polarized       = false;
};

template <typename Float>
struct spectrum_traits<Color<Float, 3>> {
    using Scalar                             = Color<dr::scalar_t<Float>, 3>;
    using Wavelength                         = Color<Float, 0>;
    using Unpolarized                        = Color<Float, 3>;
    static constexpr bool is_monochromatic   = false;
    static constexpr bool is_rgb             = true;
    static constexpr bool is_spectral        = false;
    static constexpr bool is_polarized       = false;
};

template <typename Float, size_t Size>
struct spectrum_traits<Spectrum<Float, Size>> {
    using Scalar                             = Spectrum<dr::scalar_t<Float>, Size>;
    using Wavelength                         = Spectrum<Float, Size>;
    using Unpolarized                        = Spectrum<Float, Size>;
    static constexpr bool is_monochromatic   = false;
    static constexpr bool is_rgb             = false;
    static constexpr bool is_spectral        = true;
    static constexpr bool is_polarized       = false;
};

template <typename T>
struct spectrum_traits<MuellerMatrix<T>> : spectrum_traits<T> {
    using Scalar                             = MuellerMatrix<typename spectrum_traits<T>::Scalar>;
    using Unpolarized                        = T;
    static constexpr bool is_polarized       = true;
};

template <>
struct spectrum_traits<void> {
    using Scalar      = void;
    using Wavelength  = void;
    using Unpolarized = void;
};

template <typename T>
struct spectrum_traits<dr::detail::MaskedArray<T>> : spectrum_traits<T> {
    using Scalar       = dr::detail::MaskedArray<typename spectrum_traits<T>::Scalar>;
    using Wavelength   = dr::detail::MaskedArray<typename spectrum_traits<T>::Wavelength>;
    using Unpolarized  = dr::detail::MaskedArray<typename spectrum_traits<T>::Unpolarized>;
};

NAMESPACE_END(detail)

template <typename T> constexpr bool is_monochromatic_v = detail::spectrum_traits<T>::is_monochromatic;
template <typename T> constexpr bool is_rgb_v = detail::spectrum_traits<T>::is_rgb;
template <typename T> constexpr bool is_spectral_v = detail::spectrum_traits<T>::is_spectral;
template <typename T> constexpr bool is_polarized_v = detail::spectrum_traits<T>::is_polarized;
template <typename T> using scalar_spectrum_t = typename detail::spectrum_traits<T>::Scalar;
template <typename T> using wavelength_t = typename detail::spectrum_traits<T>::Wavelength;
template <typename T> using unpolarized_spectrum_t = typename detail::spectrum_traits<T>::Unpolarized;

//! @}
// =============================================================

NAMESPACE_BEGIN(detail)

/// Type trait to strip away dynamic/masking-related type wrappers
template <typename T> struct underlying {
    using type = dr::expr_t<T>;
};

template <> struct underlying<void> {
    using type = void;
};

template <typename T> struct underlying<dr::detail::MaskedArray<T>> {
    using type = typename underlying<T>::type;
};

template <typename T, size_t Size> struct underlying<Color<T, Size>> {
    using type = Color<typename underlying<T>::type, Size>;
};

template <typename T, size_t Size> struct underlying<Spectrum<T, Size>> {
    using type = Spectrum<typename underlying<T>::type, Size>;
};

template <typename T> struct underlying<MuellerMatrix<T>> {
    using type = MuellerMatrix<typename underlying<T>::type>;
};

NAMESPACE_END(detail)

template <typename T> using underlying_t = typename detail::underlying<T>::type;

/// A variable that always evaluates to false (useful for static_assert)
template <typename... > constexpr bool false_v = false;

// =============================================================
//! @{ \name Backend capability traits
// =============================================================

/**
 * \brief Does this variant trace rays on a GPU?
 *
 * True for every backend whose \ref SceneAccel is a GPU acceleration structure
 * (OptiX, Metal, HIP-RT), false for the CPU ones (Embree / native). That is the
 * question a dozen sites in the renderer are actually asking when they refuse a
 * scalar-index accessor, skip a host-side pointer fixup, or pick the
 * non-vectorized branch.
 *
 * They used to spell it \c is_cuda_v<Float> || is_metal_v<Float>, or its
 * negation. Written that way the predicate silently answers "no" for any GPU
 * backend added afterwards -- Metal once, HIP again -- and it answers wrongly
 * in the direction that compiles and runs, taking a CPU path on a device where
 * host pointers are meaningless.
 *
 * \remark This is NOT a general replacement for asking about a specific
 * backend. Sites that select a per-backend *resource* (the color-space tables
 * in spectrum.h) or build a *discriminating* identifier (\ref type_suffix in
 * simd.h) need one real arm per backend; using this trait there would silently
 * collide two backends onto one answer, which is the same class of bug pointed
 * the other way.
 */
template <typename T>
constexpr bool is_gpu_v =
    dr::is_cuda_v<T> || dr::is_metal_v<T> || dr::is_hip_v<T>;

//! @}
// =============================================================

NAMESPACE_END(mitsuba)
