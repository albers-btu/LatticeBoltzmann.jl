using Preferences
# One process is one DIM. Absent preference => 3.
# 2D, from this project, then restart Julia:
#   using Preferences
#   set_preferences!("LatticeBoltzmann", "dim" => 2; force=true)
#   # writes gitignored LocalPreferences.toml; next process recompiles
# Delete that file to return to 3. Do not read ENV. Do not pass
# disable_invalidation=true. 3D tests and 2D tests are two processes.
_dim_pref = @load_preference("dim", 3)
_dim_pref isa Integer || error("LatticeBoltzmann dim preference must be an Int, got $(typeof(_dim_pref))")
const DIM = Int(_dim_pref)
DIM == 2 || DIM == 3 || error("LatticeBoltzmann DIM must be 2 or 3, got $DIM")

@static if DIM == 2
    const SCHEME   = :D2Q9
    const SCHEME_T = :D2Q5
else
    const SCHEME   = :D3Q19
    const SCHEME_T = :D3Q7
end

# Compile-time switches

# SRT = Single Relaxation Time
# TRT = Two Relaxation Times
const TRT                    = true # false evaluates as SRT
const KBC                    = true # entropic stabilizer; shear at ω(ν), ghosts at γω

const SURFACE                = true
const VOLUME_FORCE           = true
const EQUILIBRIUM_BOUNDARIES = true
const MOVING_BOUNDARIES      = true
const FORCE_FIELD            = true
const TEMPERATURE            = true

const UPDATE_FIELDS          = SURFACE
const APPLY_FORCE            = VOLUME_FORCE || FORCE_FIELD || TEMPERATURE

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end