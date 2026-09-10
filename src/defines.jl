const SURFACE      = true
const VOLUME_FORCE = true

const UPDATE_FIELDS = SURFACE

@static if SURFACE && !VOLUME_FORCE
    @warn "SURFACE without VOLUME_FORCE: gravity/fx,fy,fz will be ignored"
end