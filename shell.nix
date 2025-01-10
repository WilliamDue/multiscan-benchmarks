# Run with `nix-shell cuda-fhs.nix`
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; config.cudaSupport = true; } }:
let lib = pkgs.lib;
in
(pkgs.buildFHSUserEnv {
  name = "cuda-env";
  targetPkgs = pkgs: with pkgs; [ 
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
    cudatoolkit
    # futhark
    cudaPackages.cuda_cudart
    cudaPackages.cuda_nvcc
    (lib.getLib cudaPackages.cuda_nvrtc)
    (lib.getDev cudaPackages.cuda_nvrtc)
    (lib.getLib cudaPackages.cuda_cudart)
    (lib.getDev cudaPackages.cuda_cudart)
    (lib.getStatic cudaPackages.cuda_cudart)
    linuxPackages.nvidia_x11
    libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    ncurses5
    stdenv.cc
    binutils
  ];
  multiPkgs = pkgs: with pkgs; [ zlib ];
  runScript = "bash";
  profile = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export CPATH=${pkgs.cudatoolkit}/include
    export LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${lib.getStatic pkgs.cudaPackages.cuda_cudart}/lib
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${lib.getStatic pkgs.cudaPackages.cuda_cudart}/lib
  '';
}).env
