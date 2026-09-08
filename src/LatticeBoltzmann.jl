module LatticeBoltzmann

include("weights.jl")
export WEIGHTS

include("velocities.jl")
export VELOCITIES

include("memory.jl")
export Memory, MemoryContainer
export attach

include("domain.jl")
export Domain
export ρ, u
export get_N

include("model.jl")
export Model
export get_D
export run

end
