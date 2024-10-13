{
  description = "OpenCV project using Nix Flakes";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    pkgs = import nixpkgs {
      system = "x86_64-linux"; # Adjust this for your system if needed
      config.allowUnfree = true;
      config.cudaSupport = true;
      # config.cudaVersion = "12";
    };
    opencvGtk-py = pkgs.python311Packages.opencv4.override (old: {enableGtk3 = true;});
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {

      shellHook = ''
        export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:${pkgs.linuxPackages.nvidia_x11}/lib
        export CUDA_PATH=${pkgs.cudatoolkit}
        source ./.venv/bin/activate
      '';
      buildInputs = with pkgs; [
        cudaPackages.cudatoolkit 
          # linuxPackages.nvidia_x11
        cudaPackages.cudnn
        # linuxPackages.nvidia_x11
        (pkgs.writeShellScriptBin "python" ''
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
          exec ${pkgs.python3}/bin/python "$@"
        '')
        python311
        python311Packages.pip
        # python311Packages.torch-bin
        # python311Packages.torchvision-bin
        # python311Packages.transformers
        python3Packages.virtualenv
        # python311Packages.opencv4
        # python311Packages.numpy
        # python311Packages.numba
        # python311Packages.tqdm
        # python311Packages.scipy
        # python311Packages.scikitlearn
        # python311Packages.matplotlib
        # opencvGtk-py
        ffmpeg
        nvtop
      ];
    };
  };
}
