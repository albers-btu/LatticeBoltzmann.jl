using Unitful
using Unitful: Length, Velocity, Density, KinematicViscosity, Acceleration


const R_GAS = 8.314462618       # J/mol/K
const σ_SB  = 5.670374419e-8    # W/m²/K⁴


struct Units{T<:AbstractFloat}
    m::T                # Metre per cell
    kg::T               # Kilograms per lattice mass
    s::T                # Seconds per step
    K::T                # Kelvin per lattice T
    cp::T               # Specific heat capacity in J/(kg*K) (SI unit)
end

# Return the numerical cₚ value
_cp_si(cp) = cp isa Quantity ? ustrip(u"J/kg/K", cp) : cp

# How many cells span physical length si_x?
# What lattice speed should equal physical si_u? (Usually around Ma ≈ 0.05, with Ma = u/cₛ and cₛ = 1/√3 for D3Q19)
# Example: 1 mm in 100 cells, 1 m/s in 0.05 lattice u -> 5×10⁻⁷ physical seconds per time step
function Units(
       x,    u,    ρ,   # Lattice units
    si_x, si_u, si_ρ;   # SI units
    T::Type{<:AbstractFloat}=Float32, K=1, cp=1
)
    m  = T(si_x / x)
    kg = T(si_ρ / ρ) * m^3
    s  = T(u / si_u) * m
    Units{T}(m, kg, s, T(K), T(_cp_si(cp)))
end

function Units(
    si_x::Length, si_u::Velocity, si_ρ::Density;
    x, u=0.05, ρ=1, T::Type{<:AbstractFloat}=Float32, K=1, cp=1
)
    Units(x, u, ρ, ustrip(u"m", si_x), ustrip(u"m/s", si_u), ustrip(u"kg/m^3", si_ρ); T, K, cp)
end

si_x(U::Units, x)                        = x * U.m
si_t(U::Units, t)                        = t * U.s
si_u(U::Units, u)                        = u * U.m / U.s
si_ρ(U::Units, ρ)                        = ρ * U.kg / U.m^3
si_p(U::Units{T}, ρ) where {T}           = si_ρ(U, ρ) * (U.m / U.s)^2 / T(3)                            # p = ρ * c_s², c_s²=1/3
si_ν(U::Units, ν)                        = ν * U.m^2 / U.s
si_g(U::Units, g)                        = g * U.m / U.s^2
si_σ(U::Units, σ)                        = σ * U.kg / U.s^2
si_σT(U::Units, σT_lattice)              = σT_lattice * U.kg / (U.s^2 * U.K)
si_T(U::Units, T_lattice)                = T_lattice * U.K
si_Q(U::Units, Q_lattice, ρ_lattice=1)   = si_ρ(U, ρ_lattice) * U.cp * Q_lattice * U.K / U.s            # volumetric source ġ = ρ * cp * dT/dt, [ġ] = W/m³. Q_lattice is lattice dT/step
si_q(U::Units, q_lattice, ρ_lattice=1)   = si_ρ(U, ρ_lattice) * U.cp * q_lattice * U.K * U.m / U.s      # wall heat flux q = ρ * cp * (k * ∇T), [q] = W/m². q_lattice is lattice TYPE_H flux
si_h(U::Units, h_lattice, ρ_lattice=1)   = h_lattice * si_ρ(U, ρ_lattice) * U.cp * U.m / U.s
si_S(U::Units, S_lattice, ρ_lattice=1)   = S_lattice * si_ρ(U, ρ_lattice) / U.s
si_enthalpy(U::Units, H_lattice, ρ_lattice=1)   = si_ρ(U, ρ_lattice) * U.cp * U.K * U.m^3 * H_lattice
si_mass(U::Units, M_lattice, ρ_lattice=1)= si_ρ(U, ρ_lattice) * U.m^3 * M_lattice

lbm_x(U::Units, si_x)                    = si_x / U.m
lbm_t(U::Units, si_t)                    = si_t / U.s
lbm_u(U::Units, si_u)                    = si_u * U.s / U.m
lbm_u(U::Units, v::Velocity)             = lbm_u(U, ustrip(u"m/s", v))
lbm_ρ(U::Units, si_ρ)                    = si_ρ * U.m^3 / U.kg
lbm_ν(U::Units, si_ν)                    = si_ν * U.s / U.m^2
lbm_ν(U::Units, ν::KinematicViscosity)   = lbm_ν(U, ustrip(u"m^2/s", ν))
lbm_T(U::Units, Tsi)                     = Tsi / U.K
lbm_T(U::Units, θ::Quantity)             = lbm_T(U, ustrip(u"K", θ))
lbm_α(U::Units, si_α)                    = si_α * U.s / U.m^2
lbm_α(U::Units, α::Quantity)             = lbm_α(U, ustrip(u"m^2/s", α))

# Glossary
# αT    dα/dT
# kT    dk/dT
# model-α is defined as twice the physical/lattice α
function lbm_αT(U::Units, kT, ρ_lattice=1)                                                      # [dk/dT] = W/m/K² -> d(Model-α_lattice)/dT_lattice, Model-α = 2 * k/(ρ * cp)
    kT_f64 = kT isa Quantity ? ustrip(u"W/m/K^2", kT) : Float64(kT)
    αT_si = 2 * kT_f64 / (si_ρ(U, ρ_lattice) * U.cp)
    return lbm_α(U, αT_si) * U.K
end

function lbm_νT(U::Units, νT)                                                                   # [dν/dT_si] = m²/s/K -> dν_lattice/dT_lattice
    νT_f64 = νT isa Quantity ? ustrip(u"m^2/s/K", νT) : Float64(νT)
    return lbm_ν(U, νT_f64) * U.K
end

lbm_g(U::Units, si_g)                       = si_g * U.s^2 / U.m                                # Lattice gravity; fz if ρ_lbm = 1
lbm_g(U::Units, g::Acceleration)            = lbm_g(U, ustrip(u"m/s^2", g))
lbm_σ(U::Units, si_σ)                       = si_σ * U.s^2 / U.kg
lbm_σ(U::Units, σ::Quantity)                = lbm_σ(U, ustrip(u"N/m", σ))
lbm_σT(U::Units, si_σT)                     = si_σT * U.K * U.s^2 / U.kg                        # dσ/dT: N/(m·K) -> lattice σ per lattice T
lbm_σT(U::Units, σT::Quantity)              = lbm_σT(U, ustrip(u"N/m/K", σT))
lbm_Q(U::Units, si_Q, ρ_lattice=1)          = si_Q / (si_ρ(U, ρ_lattice) * U.cp * U.K / U.s)
lbm_Q(U::Units, Q::Quantity, ρ_lattice=1)   = lbm_Q(U, ustrip(u"W/m^3", Q), ρ_lattice)
lbm_q(U::Units, si_q, ρ_lattice=1)          = si_q / (si_ρ(U, ρ_lattice) * U.cp * U.K * U.m / U.s)
lbm_q(U::Units, q::Quantity, ρ_lattice=1)   = lbm_q(U, ustrip(u"W/m^2", q), ρ_lattice)
lbm_h(U::Units, hsi, ρ_lattice=1)           = hsi * U.s / (si_ρ(U, ρ_lattice) * U.cp * U.m)     # Robin h [W/m²/K] -> lattice. q_lattice = h_lat ΔT_lat
lbm_h(U::Units, h::Quantity, ρ_lattice=1)   = lbm_h(U, ustrip(u"W/m^2/K", h), ρ_lattice)

function lbm_γ(U::Units, cpT)                                                                   # dcp/dT [J/kg/K²] -> γ = d(cp/cp_ref)/dT_lat so cp/cp_ref = 1 + γ (T_lat - 1)
    cpT_f64 = cpT isa Quantity ? ustrip(u"J/kg/K^2", cpT) : Float64(cpT)
    return cpT_f64 * U.K / U.cp
end

lbm_S(U::Units, si_S, ρ_lattice=1)          = si_S * U.s / si_ρ(U, ρ_lattice)                   # Volumetric mass source kg/m³/s -> lattice Δmass/(ρ Δt)
lbm_S(U::Units, S::Quantity, ρ_lattice=1)   = lbm_S(U, ustrip(u"kg/m^3/s", S), ρ_lattice)
lbm_s(U::Units, si_s, ρ_lattice=1)          = si_s * U.s / (si_ρ(U, ρ_lattice) * U.m)           # Surface mass flux kg/m²/s -> lattice Δmass/(ρ Δt) if deposited in one cell
lbm_s(U::Units, s::Quantity, ρ_lattice=1)   = lbm_s(U, ustrip(u"kg/m^2/s", s), ρ_lattice)
lbm_Λ(U::Units, L)                          = L / (U.cp * U.K)                                  # Latent heat L [J/kg] -> lattice Λ = L/(cp K) (temperature units)
lbm_Λ(U::Units, L::Quantity)                = lbm_Λ(U, ustrip(u"J/kg", L))

# Q_lattice = C_rad (T_lattice^4 - T_∞^4) for a one-cell surface, q = ε σ T^4
function lbm_rad(U::Units{T}, ε, ρ_lattice=1) where {T}
    ε_f64 = ε isa Quantity ? ustrip(ε) : Float64(ε)
    C = ε_f64 * σ_SB * Float64(U.K)^3 * Float64(U.s) /
        (Float64(si_ρ(U, ρ_lattice)) * Float64(U.cp) * Float64(U.m))
    return T(C)
end

# Hertz–Knudsen / Clausius–Clapeyron lattice scalars
# Λ_v = L_v/(cp K), β_v = L_v/(R_sp K), p0_lat, C_hk for ṁ_lat = C_hk p_lat/√T
function lbm_evap(U::Units{T}, L_v, M, p0) where {T}
    Lv = L_v isa Quantity ? ustrip(u"J/kg", L_v) : Float64(L_v)
    Mv = M isa Quantity ? ustrip(u"kg/mol", M) : Float64(M)
    p0s = p0 isa Quantity ? ustrip(u"Pa", p0) : Float64(p0)
    Rsp = R_GAS / Mv                                                # Specific gas constant
    Λ_v = T(Lv / (U.cp * U.K))
    β_v = T(Lv / (Rsp * U.K))
    p0l = T(p0s * U.s^2 / (si_ρ(U, one(T)) * U.m^2))
    C_hk = T(U.m / (U.s * sqrt(2 * π * Rsp * U.K)))
    return Λ_v, β_v, p0l, C_hk
end

function Base.show(io::IO, U::Units)
    print(io, "Units: 1 cell = ", 1000*U.m, " mm, 1 step = ", U.s, " s")
end

Units{T}() where {T} = Units{T}(one(T), one(T), one(T), one(T), one(T))