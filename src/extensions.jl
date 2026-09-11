const TRT                    = true # false -> keep SRT

const SURFACE                = false
const VOLUME_FORCE           = true
const EQUILIBRIUM_BOUNDARIES = false
const MOVING_BOUNDARIES      = false
const FORCE_FIELD            = false
const TEMPERATURE            = true # D3Q7 thermal; TYPE_T Dirichlet; Boussinesq on fx,fy,fz

const UPDATE_FIELDS          = SURFACE
const APPLY_FORCE            = VOLUME_FORCE || FORCE_FIELD || TEMPERATURE

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end