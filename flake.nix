{
  description = "Mitsuba 3 — NixOS dev shell for building and testing scalar/llvm/cuda variants";

  inputs = {
    # Pin this to whatever you normally track. Adjust freely.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # HIP's NVIDIA-platform headers (nvidia_detail/). These are NOT in
    # ROCm/HIP and NOT in any nixpkgs ROCm output -- AMD split the NVIDIA
    # backend into its own repository, which is why searching the HIP tree for
    # them comes up empty.
    #
    # The tag MUST match the clr version in nixpkgs (currently 7.2.3).
    # Mismatched versions fail deep inside the AMD-side header -- 6.2.0 against
    # clr 7.2.3 dies on an undefined `hipHostAllocDefault`, which reads like a
    # packaging fault rather than version skew.
    hipother = {
      url = "github:ROCm/hipother/rocm-7.2.0";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, hipother }:
    let
      system = "x86_64-linux";

      # No allowUnfree / cudaSupport needed — see the note in the shellHook.
      pkgs = import nixpkgs { inherit system; };

      # Dr.Jit dlopens "libLLVM.so" at runtime for the llvm_* variants.
      # Bump this if you want a different LLVM.
      llvmLib = pkgs.llvmPackages_19.libllvm.lib;

      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        pytest      # test runner
        numpy       # used by ~53 test files
        nbformat    # test_tutorials.py
        typing-extensions
      ]);

      # nanothread does find_library(LIBATOMIC NAMES libatomic.so libatomic.so.1)
      # for GCC's 16-byte CAS. On NixOS that lives in GCC's `lib` output rather
      # than any standard search path, so CMake needs to be pointed at it.
      gccLib = pkgs.stdenv.cc.cc.lib;

      # --- HIP backend toolchain (PLAN.md §0.3) --------------------------------
      #
      # Kept in a *separate* devShell so the default one above stays free of
      # unfree packages. NVRTC is the only unfree component and it is needed
      # solely for the dual-validation harness's execution path.
      unfreePkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      rocm         = unfreePkgs.rocmPackages;
      hipClr       = rocm.clr;                 # provides bin/hipcc
      hipDeviceLib = rocm.rocm-device-libs;    # amdgcn/bitcode, else hipcc errors
      hipRt        = rocm.hiprt;               # HIP-RT: confirmed gfx90a (§0.2)
      # clang-offload-bundler, for confirming a code object really carries the
      # requested target. Offload bundles are compressed, so the target string
      # is not findable by grep -- the bundler is the only reliable check.
      rocmClang    = rocm.llvm.clang-unwrapped;
      # llvm-objcopy lives here, not in clang-unwrapped; the bundler shells out
      # to it and fails with "unable to find 'llvm-objcopy' in path" without it.
      rocmLlvm     = rocm.llvm.llvm;
      nvrtcLib     = unfreePkgs.cudaPackages.cuda_nvrtc.lib;
      nvrtcInc     = unfreePkgs.cudaPackages.cuda_nvrtc.include;

      # --- HIP-on-CUDA (PLAN.md §3.5) ----------------------------------------
      #
      # Lets the REAL HIP API (hipMalloc, hipModuleLaunchKernel, ...) compile
      # with nvcc and run on an NVIDIA GPU, so Phase 1 can be written and
      # tested without an MI210. Needs four ingredients, none obvious:
      #
      #   1. hipother          -- the nvidia_detail/ headers (flake input)
      #   2. cuda_cudart       -- cuda_runtime.h
      #   3. cuda_profiler_api -- cuda_profiler_api.h, pulled in by
      #                           nvidia_hip_runtime_api.h
      #   4. cuda_cccl         -- nv/target, pulled in by cuda_fp16.h
      #
      # Each was discovered only by following the include chain one failure at
      # a time; the error messages do not suggest the package names.
      cudaRt       = unfreePkgs.cudaPackages.cuda_cudart;
      cudaProf     = unfreePkgs.cudaPackages.cuda_profiler_api.include;
      cudaCccl     = unfreePkgs.cudaPackages.cuda_cccl;
      nvcc         = unfreePkgs.cudaPackages.cuda_nvcc;
    in
    {
      devShells.${system} = {

      default = pkgs.mkShell {
        name = "mitsuba3";

        nativeBuildInputs = with pkgs; [
          cmake
          ninja
          git
          pythonEnv
        ];

        # libpng / libjpeg-turbo are vendored under ext/ and need nothing here.
        # zlib is the exception: ext/openexr's own OpenEXRSetup.cmake does an
        # unconditional find_package(ZLIB), so a system zlib must be present even
        # though Mitsuba itself uses the vendored copy.
        buildInputs = with pkgs; [ zlib ];

        shellHook = ''
          # --- Runtime library resolution -------------------------------------
          #
          # Both GPU-ish backends are resolved by dlopen at *runtime*, not linked
          # at build time:
          #
          #   DRJIT_DYNAMIC_CUDA=ON  -> drjit-core dlopens "libcuda.so"
          #   DRJIT_DYNAMIC_LLVM=ON  -> drjit-core dlopens "libLLVM.so"
          #
          # (Verified in ext/drjit/ext/drjit-core/CMakeLists.txt:318-330 — the
          # find_package(CUDA/LLVM) branches are the *else* of those options.)
          #
          # Consequence: no CUDA toolkit and no LLVM dev package are required to
          # build, which is why this flake needs no unfree packages. We only have
          # to make the two shared objects findable at run time.
          #
          # libcuda.so comes from the NixOS driver, not nixpkgs.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${llvmLib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          # So nanothread's find_library(LIBATOMIC ...) succeeds — see gccLib above.
          export CMAKE_LIBRARY_PATH="${gccLib}/lib''${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"

          # --- Allow -march=native --------------------------------------------
          #
          # Mitsuba, Dr.Jit and struct-jit each default their *_NATIVE_FLAGS cache
          # variable to -march=native (shared ext/cmake-defaults). The Nix cc-wrapper
          # strips -m*=native when NIX_ENFORCE_NO_NATIVE_<salt>=1, which breaks
          # ext/struct-jit/src/half.cpp: it uses F16C intrinsics (_mm_cvtps_ph) that
          # then fail with "inlining failed in call to always_inline ... target
          # specific option mismatch".
          #
          # Turning the guard off fixes all three at once, rather than overriding
          # MI_NATIVE_FLAGS / DRJIT_NATIVE_FLAGS / SJIT_NATIVE_FLAGS individually.
          #
          # Tradeoff: this makes the build impure — binaries are tuned to *this*
          # CPU and are not portable to a different microarchitecture. That is the
          # right call for a local dev/baseline shell (and matches what upstream
          # does on an ordinary Linux box), but do not reuse these artifacts as if
          # they were a reproducible build.
          export NIX_ENFORCE_NO_NATIVE=0
          export NIX_ENFORCE_NO_NATIVE_${pkgs.stdenv.cc.suffixSalt}=0

          echo "mitsuba3 dev shell"
          echo "  cmake   $(cmake --version | head -1 | cut -d' ' -f3)"
          echo "  python  $(python3 --version | cut -d' ' -f2)"

          if [ -e /run/opengl-driver/lib/libcuda.so ]; then
            echo "  libcuda ok (cuda_* variants available)"
          else
            echo "  libcuda MISSING — cuda_* variants will fail at runtime"
          fi

          if [ -e "${llvmLib}/lib/libLLVM.so" ]; then
            echo "  libLLVM ok (llvm_* variants available)"
          else
            echo "  libLLVM.so not found under ${llvmLib}/lib — check the"
            echo "          llvmPackages_* attribute in flake.nix"
          fi

          cat <<'EOF'

  Build (narrow set — both reference backends, fast):
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DMI_DEFAULT_VARIANTS="scalar_rgb,llvm_ad_rgb,cuda_ad_rgb"
    cmake --build build -j 12

  MI_DEFAULT_VARIANTS regenerates build/mitsuba.conf from the template
  (CMakeLists.txt:112-128), so no conf file needs to be written by hand.
  It is only consulted when build/mitsuba.conf does not already exist —
  delete that file (or the build dir) to change the variant set.

  The full default set is:
    scalar_rgb, scalar_spectral, cuda_ad_rgb, llvm_ad_rgb, llvm_ad_spectral

  Then, in each new shell:
    source build/setpath.sh

  Test:
    pytest src -x -q                       # full suite
    pytest src/render/tests -q              # render tests only
    pytest src -q -k "not optixdenoiser"    # skip denoiser (no AMD analog)

EOF
        '';
      };

      # --- HIP backend development shell (PLAN.md §0.3) -----------------------
      #
      #   nix develop .#hip
      #
      # Provides both arms of the dual-validation harness:
      #
      #   hipcc --offload-arch=gfx90a --genco   -> valid for the REAL target?
      #   NVRTC + libcuda                       -> are the NUMBERS right?
      #
      # Neither arm requires AMD hardware. See PLAN.md §0.2 for the measurements
      # that established this and §0.3 for how the two arms are used together.
      hip = unfreePkgs.mkShell {
        name = "drjit-hip";

        # rocmClang is on PATH, not just referenced by absolute path:
        # clang-offload-bundler shells out to llvm-objcopy and fails with
        # "unable to find 'llvm-objcopy' in path" otherwise.
        nativeBuildInputs = (with unfreePkgs; [ cmake ninja git ])
          ++ [ rocmClang rocmLlvm nvcc ];

        shellHook = ''
          export HIPCC="${hipClr}/bin/hipcc"
          export HIP_PATH="${hipClr}"
          export HIP_DEVICE_LIB_PATH="${hipDeviceLib}/amdgcn/bitcode"
          export HIPRT_PATH="${hipRt}"
          export HIP_BUNDLER="${rocmClang}/bin/clang-offload-bundler"
          export HIP_OBJDUMP="${rocmLlvm}/bin/llvm-objdump"
          export NVRTC_INCLUDE="${nvrtcInc}/include"
          export NVRTC_LIB="${nvrtcLib}/lib"
          export HIP_TARGET_ARCH="gfx90a"

          # --- HIP-on-CUDA: build the real HIP API against nvcc ---------------
          #
          # Compile with:
          #   nvcc -x cu -D__HIP_PLATFORM_NVIDIA__ $HIPNV_CFLAGS <src> $HIPNV_LDFLAGS
          #
          # This exercises hipMalloc / hipLaunchKernelGGL / hipMemcpy for real,
          # which the CUDA shim deliberately does not. See §3.5 for the limits:
          # on NVIDIA, HIP is a header-level translation to CUDA, so this
          # validates API USAGE (signatures, argument order, flags, error
          # handling) and says nothing about AMD behaviour.
          export HIPNV_INCLUDE="${hipother}/hipnv/include"
          export HIPNV_CFLAGS="-D__HIP_PLATFORM_NVIDIA__ -diag-suppress 1056 -I${hipother}/hipnv/include -I${hipClr}/include -I${cudaRt}/include -I${cudaProf}/include -I${cudaCccl}/include -I${nvrtcInc}/include"
          # -lcuda is NOT optional: hipModule* lowers to the CUDA DRIVER API
          # (cuLaunchKernel, cuCtxDestroy_v2, ...), not the runtime API, so
          # linking only -lcudart fails with undefined cu* symbols.
          export HIPNV_LDFLAGS="-L${cudaRt}/lib -lcudart -L${nvrtcLib}/lib -l:libnvrtc.alt.so.12 -L/run/opengl-driver/lib -lcuda"
          export CUDART_LIB="${cudaRt}/lib"

          # libcuda comes from the NixOS driver; NVRTC from nixpkgs.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${nvrtcLib}/lib:${cudaRt}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          # hipcc cannot find the device bitcode on NixOS without this.
          export HIPCC_COMPILE_FLAGS_APPEND="--rocm-device-lib-path=$HIP_DEVICE_LIB_PATH"

          echo "drjit HIP dev shell — target $HIP_TARGET_ARCH"
          [ -x "$HIPCC" ] && echo "  hipcc   ok" || echo "  hipcc   MISSING"
          [ -d "$HIP_DEVICE_LIB_PATH" ] && echo "  devlibs ok" || echo "  devlibs MISSING"
          [ -e "$HIPRT_PATH/include/hiprt/hiprt.h" ] && echo "  hiprt   ok (gfx90a supported — PLAN §0.2)" || echo "  hiprt   MISSING"
          [ -e "$NVRTC_LIB/libnvrtc.alt.so.12" ] && echo "  nvrtc   ok" || echo "  nvrtc   MISSING"
          [ -e /run/opengl-driver/lib/libcuda.so ] && echo "  libcuda ok (execution arm available)" || echo "  libcuda MISSING — execution arm unavailable"

          cat <<'EOF'

  Build the harness (from the drjit-core checkout). It is standalone --
  it does NOT build drjit-core, and adds no upstream seam:
    cmake -S tools/hip_validate -B build-hip -G Ninja
    cmake --build build-hip

  Run it on a kernel:
    ./build-hip/hip_validate tools/hip_validate/kernels/smoke_arith.hip

  Both arms run by default; --no-exec or --no-gfx skips one.
  See tools/hip_validate/README.md for the kernel contract.

EOF
        '';
      };

      };
    };
}
