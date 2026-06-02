{
  description = "Eigen3ToPython: python bindings between eigen and numpy";

  inputs.mc-rtc-nix.url = "github:mc-rtc/nixpkgs";

  outputs =
    inputs:
    inputs.mc-rtc-nix.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        pyOverrideAttrs.eigen3-to-python = {
          src = lib.cleanSource ./.;
        };
      }
    );
}
