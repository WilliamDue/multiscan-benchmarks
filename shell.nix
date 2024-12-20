# Run with `nix-shell cuda-fhs.nix`
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
(pkgs.buildFHSUserEnv {
  name = "cuda-env";
  targetPkgs = (pkgs: with pkgs; [ 
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    util-linux
    m4
    gperf
    unzip
    linuxPackages.nvidia_x11
    libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    ncurses5
    # stdenv.cc
    # futhark
    gcc12
    binutils
    ispc
    cudaPackages.cudatoolkit
    cudaPackages.markForCudatoolkitRootHook
  ]);
  multiPkgs = pkgs: with pkgs; [ zlib ];
  runScript = "bash";
  profile = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LIBRARY_PATH=${pkgs.cudatoolkit}/lib
    export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';
}).env
