{
  description = "Eigen3ToPython: python bindings between eigen and numpy";

  inputs.mc-rtc-nix.url = "github:mc-rtc/nixpkgs";

  outputs =
    inputs:
    inputs.mc-rtc-nix.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        pyOverrideAttrs.eigen3-to-python =
          { pkgs-final, ... }:
          {
            src = lib.cleanSource ./.;

            # Override the default CMake install step
            installPhase = ''
              runHook preInstall

              # 1. Define the destination directory inside the Nix store output
              # Using python.sitePackages handles the "lib/python3.13/site-packages" string automatically
              local targetDir="$out/${pkgs-final.python3Packages.python.sitePackages}/eigen"
              mkdir -p "$targetDir"

              # 2. Copy your built files from the build tree to the target store path
              # (Adjust 'build/python/Release/eigen/' path if your build folder structure differs slightly)
              # TODO: changed here, use upstream
              cp -r python3/Release/eigen/* "$targetDir"

              runHook postInstall
            '';
          };
      }
    );
}
