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
lbm_g(U::Units, si_g)                  = si_g * U.s^2 / U.m # lattice gravity; fz if ρ_lbm=1
lbm_g(U::Units, g::Acceleration)       = lbm_g(U, ustrip(u"m/s^2", g))
lbm_σ(U::Units, si_σ)                  = si_σ * U.s^2 / U.kg
lbm_σ(U::Units, σ::Quantity)           = lbm_σ(U, ustrip(u"N/m", σ))
lbm_Q(U::Units, si_Q, ρlat=1)          = si_Q / (si_ρ(U, ρlat) * U.cp * U.K / U.s)
lbm_Q(U::Units, Q::Quantity, ρlat=1)   = lbm_Q(U, ustrip(u"W/m^3", Q), ρlat)
lbm_q(U::Units, si_q, ρlat=1)          = si_q / (si_ρ(U, ρlat) * U.cp * U.K * U.m / U.s)
lbm_q(U::Units, q::Quantity, ρlat=1)   = lbm_q(U, ustrip(u"W/m^2", q), ρlat)

function Base.show(io::IO, U::Units)
    print(io, "Units: 1 cell = ", 1000*U.m, " mm, 1 step = ", U.s, " s")
end

Units{T}() where {T} = Units{T}(one(T), one(T), one(T), one(T), one(T))