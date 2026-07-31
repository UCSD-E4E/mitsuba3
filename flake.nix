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

        # Build-time deps only. zlib / libpng / libjpeg-turbo / OpenEXR are all
        # vendored under ext/, so they are deliberately absent here.
        buildInputs = [ ];

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

  Build:
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

  Then, in each new shell:
    source build/setpath.sh

  Test:
    pytest src -x -q                       # full suite
    pytest src/render/tests -q              # render tests only
    pytest src -q -k "not optixdenoiser"    # skip denoiser

  Variants: with no mitsuba.conf, CMakeLists.txt:116 defaults to
    scalar_rgb, scalar_spectral, cuda_ad_rgb, llvm_ad_rgb, llvm_ad_spectral
  To build a narrower (much faster) set, write a mitsuba.conf first.

EOF
        '';
      };
    };
}
