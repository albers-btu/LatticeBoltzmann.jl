# Compile-time switches

# SRT = Single Relaxation Time
# TRT = Two Relaxation Times
const TRT                    = true # false evaluates as SRT

const SURFACE                = true
const VOLUME_FORCE           = true
const EQUILIBRIUM_BOUNDARIES = false
const MOVING_BOUNDARIES      = false
const FORCE_FIELD            = false
const TEMPERATURE            = true

const UPDATE_FIELDS          = SURFACE
const APPLY_FORCE            = VOLUME_FORCE || FORCE_FIELD || TEMPERATURE

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end