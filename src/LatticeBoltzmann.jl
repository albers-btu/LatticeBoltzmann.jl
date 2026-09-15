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
export si_x, si_t, si_u, si_ρ, si_p, si_T, si_Q, si_q, si_σT, si_S, si_enthalpy, si_mass, si_h, lbm_ν, lbm_νT, lbm_g, lbm_σ, lbm_σT, lbm_u, lbm_T, lbm_Q, lbm_q, lbm_h, lbm_γ, lbm_Λ, lbm_evap, lbm_S, lbm_s, lbm_rad, lbm_αT, σ_SB

include("flags.jl")
export TYPE_S, TYPE_E, TYPE_T, TYPE_F, TYPE_I, TYPE_G, TYPE_H, TYPE_MS
export TYPE_BO, TYPE_IF, TYPE_IG, TYPE_GI, TYPE_SU

include("memory.jl")
export Memory, MemoryContainer
export attach

include("domain.jl")
export Domain
export ρ, u, F, τ, msrc, mp
export T, Q, htc, fs, thermal_k, thermal_k_s, thermal_k_l, enthalpy, heat_source, cell_enthalpy, metal_mass
export dissolved_D, arrhenius_k, agent_inventory
export energy_budget, reset_energy_budget!
export mass_budget, reset_mass_budget!
export get_N
export increment_time_step!

include("log.jl")
export start_run_log!, stop_run_log!

include("kernel.jl")
export initialize_kernel!, stream_collide_even_kernel!, stream_collide_odd_kernel!, powder_gas_kernel!

include("plic.jl")
export plic_cube, calculate_curvature, calculate_normal_py

include("laser.jl")
export Laser, set_laser_position!, deposit_laser!, fresnel_absorptance

include("powder.jl")
export PowderJet, set_powder_jet_position!, aim_powder_jet!, advance_powder_jet!

include("model.jl")
include("bubble.jl")
export BubbleTracker, update_bubbles!, bubble_records, set_bubble_n!
export Nucleation, nucleate_bubbles!, foam_metrics
export young_laplace_p, bubble_radius, P_ATM_LAT
export epstein_plesset_R2
export Model
export get_D
export run!, export!
export update_force_field!, reset_force_field!

end
