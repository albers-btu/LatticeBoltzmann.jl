module LatticeBoltzmann

include("weights.jl")
export WEIGHTS

include("velocities.jl")
export VELOCITIES

include("domain.jl")
export Domain

include("memory.jl")
export Memory

end
