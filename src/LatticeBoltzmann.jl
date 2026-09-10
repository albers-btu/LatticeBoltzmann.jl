module LatticeBoltzmann

include("weights.jl")
export WEIGHTS
export weights

include("velocities.jl")
export VELOCITIES
export velocities

include("extensions.jl")
export SURFACE, VOLUME_FORCE, UPDATE_FIELDS

include("units.jl")
export Units
export si_x, si_t, si_u, si_ρ, si_p, lbm_ν, lbm_g, lbm_σ, lbm_u

include("flags.jl")
export TYPE_S, TYPE_E, TYPE_T, TYPE_F, TYPE_I, TYPE_G, TYPE_MS
export TYPE_BO, TYPE_IF, TYPE_IG, TYPE_GI, TYPE_SU

include("memory.jl")
export Memory, MemoryContainer
export attach

include("domain.jl")
export Domain
export ρ, u, τ
export get_N
export increment_time_step!

include("kernel.jl")
export initialize_kernel!, stream_collide_even_kernel!, stream_collide_odd_kernel!

include("model.jl")
export Model
export get_D
export run!, export!

end
