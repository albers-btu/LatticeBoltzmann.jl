const TRT                    = false # false -> keep SRT

const SURFACE                = false
const VOLUME_FORCE           = false
const EQUILIBRIUM_BOUNDARIES = false
const MOVING_BOUNDARIES      = false

const UPDATE_FIELDS          = SURFACE

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end