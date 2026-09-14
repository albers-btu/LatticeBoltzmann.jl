using Unitful
using Unitful: Length, Velocity, Density, KinematicViscosity, Acceleration

struct Units{T<:AbstractFloat}
    m::T
    kg::T
    s::T
    K::T    # Kelvin at lattice T = 1 (identity: K=1)
    cp::T   # J/(kg·K); identity: cp=1
end

_cp_si(cp) = cp isa Quantity ? ustrip(u"J/kg/K", cp) : cp

function Units(x, u, ρ, si_x, si_u, si_ρ; T::Type{<:AbstractFloat}=Float32, K=1, cp=1)
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

si_x(U::Units, x)              = x * U.m
si_t(U::Units, t)              = t * U.s
si_u(U::Units, u)              = u * U.m / U.s
si_ρ(U::Units, ρ)              = ρ * U.kg / U.m^3
si_p(U::Units{T}, ρ) where {T} = si_ρ(U, ρ) * (U.m / U.s)^2 / T(3) # p = ρ c_s², c_s²=1/3
si_ν(U::Units, ν)              = ν * U.m^2 / U.s
si_g(U::Units, g)              = g * U.m / U.s^2
si_σ(U::Units, σ)              = σ * U.kg / U.s^2
si_T(U::Units, Tlat)           = Tlat * U.K
lbm_T(U::Units, Tsi)           = Tsi / U.K
lbm_T(U::Units, θ::Quantity)   = lbm_T(U, ustrip(u"K", θ))
# volumetric source ġ = ρ cp dT/dt  → W/m³. Qlat is lattice dT/step.
si_Q(U::Units, Qlat, ρlat=1)   = si_ρ(U, ρlat) * U.cp * Qlat * U.K / U.s
# wall heat flux q = ρ cp (k ∇T)     → W/m². qlat is lattice TYPE_H flux.
si_q(U::Units, qlat, ρlat=1)   = si_ρ(U, ρlat) * U.cp * qlat * U.K * U.m / U.s

lbm_x(U::Units, si_x)                  = si_x / U.m
lbm_t(U::Units, si_t)                  = si_t / U.s
lbm_u(U::Units, si_u)                  = si_u * U.s / U.m
lbm_u(U::Units, v::Velocity)           = lbm_u(U, ustrip(u"m/s", v))
lbm_ρ(U::Units, si_ρ)                  = si_ρ * U.m^3 / U.kg
lbm_ν(U::Units, si_ν)                  = si_ν * U.s / U.m^2
lbm_ν(U::Units, ν::KinematicViscosity) = lbm_ν(U, ustrip(u"m^2/s", ν))
# dk/dT [W/m/K²] → d(Model-α_lat)/dT_lat, Model-α = 2 k/(ρ cp)
function lbm_αT(U::Units, kT, ρlat=1)
    kTf = kT isa Quantity ? ustrip(u"W/m/K^2", kT) : Float64(kT)
    αT_si = 2 * kTf / (si_ρ(U, ρlat) * U.cp)
    return lbm_ν(U, αT_si) * U.K
end
# dν/dT_si [m²/s/K] → dν_lat/dT_lat
function lbm_νT(U::Units, νT)
    νTf = νT isa Quantity ? ustrip(u"m^2/s/K", νT) : Float64(νT)
    return lbm_ν(U, νTf) * U.K
end
lbm_g(U::Units, si_g)                  = si_g * U.s^2 / U.m # lattice gravity; fz if ρ_lbm=1
lbm_g(U::Units, g::Acceleration)       = lbm_g(U, ustrip(u"m/s^2", g))
lbm_σ(U::Units, si_σ)                  = si_σ * U.s^2 / U.kg
lbm_σ(U::Units, σ::Quantity)           = lbm_σ(U, ustrip(u"N/m", σ))
# dσ/dT: N/(m·K) → lattice σ per lattice T
lbm_σT(U::Units, si_σT)                = si_σT * U.K * U.s^2 / U.kg
lbm_σT(U::Units, σT::Quantity)         = lbm_σT(U, ustrip(u"N/m/K", σT))
si_σT(U::Units, σTlat)                 = σTlat * U.kg / (U.s^2 * U.K)
lbm_Q(U::Units, si_Q, ρlat=1)          = si_Q / (si_ρ(U, ρlat) * U.cp * U.K / U.s)
lbm_Q(U::Units, Q::Quantity, ρlat=1)   = lbm_Q(U, ustrip(u"W/m^3", Q), ρlat)
lbm_q(U::Units, si_q, ρlat=1)          = si_q / (si_ρ(U, ρlat) * U.cp * U.K * U.m / U.s)
lbm_q(U::Units, q::Quantity, ρlat=1)   = lbm_q(U, ustrip(u"W/m^2", q), ρlat)
# Robin h [W/m²/K] → lattice. qlat = h_lat ΔT_lat.
lbm_h(U::Units, hsi, ρlat=1)           = hsi * U.s / (si_ρ(U, ρlat) * U.cp * U.m)
lbm_h(U::Units, h::Quantity, ρlat=1)   = lbm_h(U, ustrip(u"W/m^2/K", h), ρlat)
si_h(U::Units, hlat, ρlat=1)           = hlat * si_ρ(U, ρlat) * U.cp * U.m / U.s
# dcp/dT [J/kg/K²] → γ = d(cp/cp_ref)/dT_lat so cp/cp_ref = 1 + γ (T_lat - 1).
function lbm_γ(U::Units, cpT)
    cpTf = cpT isa Quantity ? ustrip(u"J/kg/K^2", cpT) : Float64(cpT)
    return cpTf * U.K / U.cp
end
# volumetric mass source kg/m³/s → lattice Δmass/(ρ Δt)
lbm_S(U::Units, si_S, ρlat=1)          = si_S * U.s / si_ρ(U, ρlat)
lbm_S(U::Units, S::Quantity, ρlat=1)   = lbm_S(U, ustrip(u"kg/m^3/s", S), ρlat)
si_S(U::Units, Slat, ρlat=1)           = Slat * si_ρ(U, ρlat) / U.s
# surface mass flux kg/m²/s → lattice Δmass/(ρ Δt) if deposited in one cell
lbm_s(U::Units, si_s, ρlat=1)          = si_s * U.s / (si_ρ(U, ρlat) * U.m)
lbm_s(U::Units, s::Quantity, ρlat=1)   = lbm_s(U, ustrip(u"kg/m^2/s", s), ρlat)
# latent heat L [J/kg] → lattice Λ = L/(cp K) (temperature units)
lbm_Λ(U::Units, L)                     = L / (U.cp * U.K)
lbm_Λ(U::Units, L::Quantity)           = lbm_Λ(U, ustrip(u"J/kg", L))
# Σ cell enthalpy (lattice T·cell) → J. One cell volume is m³.
si_enthalpy(U::Units, Hlat, ρlat=1)    = si_ρ(U, ρlat) * U.cp * U.K * U.m^3 * Hlat
# Σ lattice mass (ρ_lat × cell) → kg.
si_mass(U::Units, Mlat, ρlat=1)        = si_ρ(U, ρlat) * U.m^3 * Mlat

const R_GAS = 8.314462618          # J/mol/K
const σ_SB  = 5.670374419e-8       # W/m²/K⁴

# Q_lat = C_rad (T_lat^4 - T_∞^4) for a one-cell surface, q = ε σ T^4.
function lbm_rad(U::Units{T}, ε, ρlat=1) where {T}
    εf = ε isa Quantity ? ustrip(ε) : Float64(ε)
    C = εf * σ_SB * Float64(U.K)^3 * Float64(U.s) /
        (Float64(si_ρ(U, ρlat)) * Float64(U.cp) * Float64(U.m))
    return T(C)
end

# Hertz–Knudsen / Clausius–Clapeyron lattice scalars.
# Λ_v = L_v/(cp K), β_v = L_v/(R_sp K), p0_lat, C_hk for ṁ_lat = C_hk p_lat/√T.
function lbm_evap(U::Units{T}, L_v, M, p0) where {T}
    Lv = L_v isa Quantity ? ustrip(u"J/kg", L_v) : Float64(L_v)
    Mv = M isa Quantity ? ustrip(u"kg/mol", M) : Float64(M)
    p0s = p0 isa Quantity ? ustrip(u"Pa", p0) : Float64(p0)
    Rsp = R_GAS / Mv
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