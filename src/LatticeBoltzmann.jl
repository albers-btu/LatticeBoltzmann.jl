module LatticeBoltzmann

include("weights.jl")
export WEIGHTS
export weights

include("velocities.jl")
export VELOCITIES
export velocities

include("extensions.jl")
export TRT, SURFACE, VOLUME_FORCE, UPDATE_FIELDS, EQUILIBRIUM_BOUNDARIES, MOVING_BOUNDARIES, FORCE_FIELD, TEMPERATURE

include("units.jl")
export Units
export si_x, si_t, si_u, si_ρ, si_p, si_T, lbm_ν, lbm_g, lbm_σ, lbm_u

include("flags.jl")
export TYPE_S, TYPE_E, TYPE_T, TYPE_F, TYPE_I, TYPE_G, TYPE_MS
export TYPE_BO, TYPE_IF, TYPE_IG, TYPE_GI, TYPE_SU

include("memory.jl")
export Memory, MemoryContainer
export attach

include("domain.jl")
export Domain
export ρ, u, F, τ
export T
export get_N
export increment_time_step!

include("log.jl")
export start_run_log!, stop_run_log!

include("kernel.jl")
export initialize_kernel!, stream_collide_even_kernel!, stream_collide_odd_kernel!

include("plic.jl")
export plic_cube, calculate_curvature, calculate_normal_py

include("model.jl")
export Model
export get_D
export run!, export!
export update_force_field!, reset_force_field!

end
