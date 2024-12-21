# Run with nix-shell cuda-shell.nix
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
pkgs.mkShell {
   name = "cuda-env-shell";
   buildInputs = with pkgs; [
     git gitRepo gnupg autoconf curl
     procps gnumake util-linux m4 gperf unzip
     cudatoolkit linuxPackages.nvidia_x11
     libGLU libGL
     futhark
     cudaPackages.cuda_cudart
     xorg.libXi xorg.libXmu freeglut
     xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
     ncurses5 stdenv.cc binutils
   ];
   shellHook = ''
      export CUDA_PATH=${pkgs.cudatoolkit}
      export CPATH=${pkgs.cudatoolkit}/include
      export LIBRARY_PATH=${pkgs.cudatoolkit}/lib
      export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.linuxPackages.nvidia_x11}/lib
      export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"
   '';
}
