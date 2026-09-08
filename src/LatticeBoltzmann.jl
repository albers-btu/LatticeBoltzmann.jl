module LatticeBoltzmann

include("weights.jl")
export WEIGHTS

include("velocities.jl")
export VELOCITIES

include("memory.jl")
export Memory

include("domain.jl")
export Domain

include("model.jl")
export Model
export run
export get_N, get_D

end
