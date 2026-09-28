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
const FOAM                   = false # dissolved-gas bubbles; requires SURFACE

const UPDATE_FIELDS          = SURFACE
const APPLY_FORCE            = VOLUME_FORCE || FORCE_FIELD || TEMPERATURE

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end

@static if FOAM && !SURFACE
    error("FOAM requires SURFACE")
end