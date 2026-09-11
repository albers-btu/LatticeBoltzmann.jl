const TRT                    = true # false -> keep SRT

const SURFACE                = false
const VOLUME_FORCE           = false
const EQUILIBRIUM_BOUNDARIES = true
const MOVING_BOUNDARIES      = false
const FORCE_FIELD            = true

const UPDATE_FIELDS          = SURFACE
const APPLY_FORCE            = VOLUME_FORCE || FORCE_FIELD

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end