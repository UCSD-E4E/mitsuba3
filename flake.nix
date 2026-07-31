{
  description = "Mitsuba 3 — NixOS dev shell for building and testing scalar/llvm/cuda variants";

  inputs = {
    # Pin this to whatever you normally track. Adjust freely.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
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
    in
    {
      devShells.${system}.default = pkgs.mkShell {
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
    };
}
