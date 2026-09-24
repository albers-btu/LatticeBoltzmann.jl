# Energy (enthalpy) account
# E = Q - RAD - EVAP - WALL + POWDER
const EACC_Q = 1        # Laser
const EACC_RAD = 2      # Radiation
const EACC_EVAP = 3     # Evaporation
const EACC_WALL = 4     # Wall heat
const EACC_POWDER = 5   # Powder (source) enthalpy
const EACC_N = 5        # Length of energy account array

# Mass account 
# M = M0 + POWDER − EVAP + residual.
const MACC_EVAP = 1     # Evaporation
const MACC_POWDER = 2   # Powder (source)
const MACC_N = 2        # Length of mass account array

# Represents the complete state of one grid
mutable struct Domain{
    CType<:AbstractFloat,               # Compute Type, default is Float32
    SType<:AbstractFloat,               # Store Type, default is Float32
    Aρ<:AbstractArray{CType},           # Array for ρ (density)
    Au<:AbstractArray{CType},           # Array for u (velocity)
    Afi<:AbstractArray{SType},          # Array for fᵢ DDF (discrete distribution function)
    Af<:AbstractArray{UInt8}            # Array for flags
}
    Nx::UInt                            # Lattice size x
    Ny::UInt                            # Lattice size y
    Nz::UInt                            # Lattice size z

    Ox::Int                             # Offset in x (if this is a subdomain)
    Oy::Int                             # Offset in y (if this is a subdomain)
    Oz::Int                             # Offset in z (if this is a subdomain)

    ν::CType                            # Kinematic viscosity
    N::Int                              # Number of cells
    ω::CType                            # For BGK: 1/(3ν+1/2)

    fx::CType                           # Body force per volume in x
    fy::CType                           # Body force per volume in y
    fz::CType                           # Body force per volume in z

    σ::CType                            # Surface tension at reference temperature
    σT::CType                           # dσ/dT (≠ 0 leads to Marangoni effects)
    Tσ::CType                           # Reference temperature

    ρ::Memory{CType, Aρ}                # ρ (density)
    u::Memory{CType, Au}                # u (velocity)
    F::Memory{CType, Au}                # F (force)
    fi::Memory{SType, Afi}              # fᵢ (discrete distribution function)
    flags::Memory{UInt8, Af}            # flags

    @static if SURFACE
        ϕ::Memory{CType, Aρ}            # Liquid fill fraction (0 gas, 1 liquid)
        mass::Memory{CType, Aρ}         # Liquid mass (mass ≈ ϕ*ρ)
        massex::Memory{CType, Aρ}       # Excess mass
        msrc::Memory{CType, Aρ}         # Mass source rate (fill fraction per lattice step)
        mp::Memory{CType, Aρ}           # Unmelted powder mass
        τ_p::CType                      # Powder lifetime in lattice steps
        T_p::CType                      # Powder temperature
    end

    @static if TEMPERATURE
        α::CType                        # Thermal diffusivity (mix)
        α_s::CType                      # Thermal diffusivity (solid) (twice Chapman-Enskog diffusivity)
        α_l::CType                      # Thermal diffusivity (liquid)
        α_sT::CType                     # d(α_solid ) / d(T_lattice)
        α_lT::CType                     # d(α_liquid) / d(T_lattice)
        γ_s::CType                      # Heat capacity ratio (solid): d(cp/cp_ref)/d(T_lattice); γ = 0 leads to constant cp
        γ_l::CType                      # Heat capacity ratio (liquid)
        ν_s::CType                      # Kinematic viscosity (solid)
        ν_l::CType                      # Kinematic viscosity (liquid)
        ν_sT::CType                     # d(ν_solid ) / d(T_lattice)
        ν_lT::CType                     # d(ν_liquid) / d(T_lattice)
        β::CType                        # Thermal expansion coefficient (buoyancy)
        T_avg::CType                    # Reference temperature
        ω_T::CType                      # BGK relaxation rate for gᵢ (D3Q7 heat DDF)
        T::Memory{CType, Aρ}            # Temperature
        gi::Memory{SType, Afi}          # gᵢ (discrete distribution function)
        Q::Memory{CType, Aρ}            # Volumetric heat source (ΔQ per step)
        h::Memory{CType, Aρ}            # Robin q = Q + h (T - T∞); h = 0 leads to pure Neumann
        
                                        # Melting
        Λ::CType                        # Latent heat L/cp in lattice T; 0 leads to no melting
        Ts::CType                       # Solidus temperature
        Tl::CType                       # Liquidus temperature (Tₛ = Tₗ leads to isothermal Stefan)
        K0::CType                       # Kozeny–Carman Darcy drag
        fs::Memory{CType, Aρ}           # Solid fraction (0 liquid, 1 solid)
        
                                        # Evaporation (Hertz-Knudsen)
        Λ_v::CType                      # Vaporization L_v/(cp K); 0 leads to no evaporation
        T_v::CType                      # Boiling T (lattice)
        C_hk::CType                     # Hertz-Knudsen prefactor (lattice)
        p0v::CType                      # p_atm (lattice)
        β_v::CType                      # L_v/(R_sp K) Clausius–Clapeyron
        
                                        # Radiation
                                        # Q_rad = C_rad (T⁴ - T_rad⁴)
        C_rad::CType                    # SI prefactor: ε (emissivity) σ (Stefan-Boltzman) K³ (Kelvin per lattice) s (Δt) / (ρ cp m (Δx))
        T_rad::CType                    # Far-field T for radiation (lattice)
        
        Eacc::Memory{CType, Aρ}         # Energy account
        H0::CType                       # Initial Enthalpy, H ≈ H0 + Q - RAD - EVAP - WALL + POWDER
        E_powder::CType                 # Source on the host (e.g. powder jet)

        @static if SURFACE
            Macc::Memory{CType, Aρ}     # Mass account
            M0::CType                   # Initial Mass, M ≈ M0 + POWDER - EVAP
            M_powder::CType             # Source on the host (e.g. powder jet)
        end
    end

    t::UInt64                           # Lattice time-step counter
end

function Domain(
    Nx, Ny, Nz, 
    Ox, Oy, Oz, 
    ν, 
    fx, fy, fz, 
    scheme, backend, 
    ::Type{CType}, ::Type{SType};
    σ::CType = zero(CType),
    σT::CType = zero(CType),
    Tσ::CType = one(CType),
    α::CType = zero(CType),
    α_s::CType = zero(CType),
    α_l::CType = zero(CType),
    α_sT::CType = zero(CType),
    α_lT::CType = zero(CType),
    γ_s::CType = zero(CType),
    γ_l::CType = zero(CType),
    ν_s::CType = zero(CType),
    ν_l::CType = zero(CType),
    ν_sT::CType = zero(CType),
    ν_lT::CType = zero(CType),
    β::CType = zero(CType),
    T_avg::CType = one(CType),
    Λ::CType = zero(CType),
    Ts::CType = zero(CType),
    Tl::CType = zero(CType),
    K0::CType = zero(CType),
    Λ_v::CType = zero(CType),
    T_v::CType = zero(CType),
    C_hk::CType = zero(CType),
    p0v::CType = zero(CType),
    β_v::CType = zero(CType),
    C_rad::CType = zero(CType),
    T_rad::CType = one(CType),
    τ_p::CType = zero(CType),
    T_p::CType = one(CType),
) where {CType, SType}
    nvel = length(WEIGHTS[scheme])

    N = Int(Nx) * Int(Ny) * Int(Nz)
                                                                    
    ω = one(CType) / (CType(3) * CType(ν) + CType(1) / CType(2))    # Relaxation rate ω = 1/τ
                                                                    # BGK / TRT⁺ rate
                                                                    # ω = 1 / (3⋅ν + 1/2)
    AT = arraytype(backend)

    ρ = Memory(AT{CType}(undef, N))
    fill!(ρ.data, one(CType))

    u = Memory(AT{CType}(undef, N, 3))
    fill!(u.data, zero(CType))

    F = Memory(AT{CType}(undef, N, 3))
    fill!(F.data, zero(CType))

    fi = Memory(AT{SType}(undef, N * nvel))
    fill!(fi.data, zero(SType))

    flags = Memory(AT{UInt8}(undef, N))
    fill!(flags.data, 0x00)

    @static if SURFACE
        ϕ = Memory(AT{CType}(undef, N))
        fill!(ϕ.data, zero(CType))

        mass = Memory(AT{CType}(undef, N))
        fill!(mass.data, zero(CType))

        massex = Memory(AT{CType}(undef, N))
        fill!(massex.data, zero(CType))

        msrc = Memory(AT{CType}(undef, N))
        fill!(msrc.data, zero(CType))

        mp = Memory(AT{CType}(undef, N))
        fill!(mp.data, zero(CType))
    end

    @static if TEMPERATURE
        αT = α == zero(CType) ? CType(ν) : α
        αs = α_s == zero(CType) ? αT : α_s
        αl = α_l == zero(CType) ? αT : α_l
        νs = ν_s == zero(CType) ? CType(ν) : ν_s
        νl = ν_l == zero(CType) ? CType(ν) : ν_l
        ω_T = one(CType) / (CType(2) * αT + CType(1) / CType(2))    # D3Q7 heat relaxation
                                                                    # c_sT² = 1/4
                                                                    # ω_T = 1 / (2⋅αT + 1/2)
        Tmem = Memory(AT{CType}(undef, N))
        fill!(Tmem.data, T_avg)
        gi = Memory(AT{SType}(undef, N * 7))
        fill!(gi.data, zero(SType))
        Qmem = Memory(AT{CType}(undef, N))
        fill!(Qmem.data, zero(CType))
        hmem = Memory(AT{CType}(undef, N))
        fill!(hmem.data, zero(CType))
        fsmem = Memory(AT{CType}(undef, N))
        fill!(fsmem.data, zero(CType))
        Eacc = Memory(AT{CType}(undef, EACC_N))
        fill!(Eacc.data, zero(CType))
        @static if SURFACE
            Macc = Memory(AT{CType}(undef, MACC_N))
            fill!(Macc.data, zero(CType))
        end
    end

    @static if SURFACE
        @static if TEMPERATURE
            Domain(
                UInt(Nx), UInt(Ny), UInt(Nz),
                Int(Ox), Int(Oy), Int(Oz),
                CType(ν), N, ω,
                CType(fx), CType(fy), CType(fz),
                CType(σ), CType(σT), CType(Tσ),
                ρ, u, F, fi, flags,
                ϕ, mass, massex, msrc, mp, CType(τ_p), CType(T_p),
                αT, αs, αl, CType(α_sT), CType(α_lT),
                CType(γ_s), CType(γ_l), 
                νs, νl, CType(ν_sT), CType(ν_lT), 
                β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
                Λ_v, T_v, C_hk, p0v, β_v, CType(C_rad), CType(T_rad),
                Eacc, zero(CType), zero(CType),
                Macc, zero(CType), zero(CType),
                UInt64(0)
            )
        else
            Domain(
                UInt(Nx), UInt(Ny), UInt(Nz),
                Int(Ox), Int(Oy), Int(Oz),
                CType(ν), N, ω,
                CType(fx), CType(fy), CType(fz),
                CType(σ), CType(σT), CType(Tσ),
                ρ, u, F, fi, flags,
                ϕ, mass, massex, msrc, mp,
                CType(τ_p), CType(T_p),
                UInt64(0)
            )
        end
    else
        @static if TEMPERATURE
            Domain(
                UInt(Nx), UInt(Ny), UInt(Nz),
                Int(Ox), Int(Oy), Int(Oz),
                CType(ν), N, ω,
                CType(fx), CType(fy), CType(fz),
                CType(σ), CType(σT), CType(Tσ),
                ρ, u, F, fi, flags,
                αT, αs, αl, CType(α_sT), CType(α_lT), 
                CType(γ_s), CType(γ_l), νs, νl, CType(ν_sT), CType(ν_lT), 
                β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
                Λ_v, T_v, C_hk, p0v, β_v, CType(C_rad), CType(T_rad),
                Eacc, zero(CType), zero(CType),
                UInt64(0)
            )
        else
            Domain(
                UInt(Nx), UInt(Ny), UInt(Nz),
                Int(Ox), Int(Oy), Int(Oz),
                CType(ν), N, ω,
                CType(fx), CType(fy), CType(fz),
                CType(σ), CType(σT), CType(Tσ),
                ρ, u, F, fi, flags,
                UInt64(0)
            )
        end
    end
end

get_N(domain::Domain) = Int(domain.Nx) * Int(domain.Ny) * Int(domain.Nz)

ρ(domain::Domain) = domain.ρ
u(domain::Domain) = domain.u
F(domain::Domain) = domain.F
fi(domain::Domain) = domain.fi
flags(domain::Domain) = domain.flags

@static if SURFACE
    ϕ(domain::Domain) = domain.ϕ
    mass(domain::Domain) = domain.mass
    massex(domain::Domain) = domain.massex
    msrc(domain::Domain) = domain.msrc
    mp(domain::Domain) = domain.mp
end

@static if TEMPERATURE
    T(domain::Domain) = domain.T
    gi(domain::Domain) = domain.gi
    Q(domain::Domain) = domain.Q
    htc(domain::Domain) = domain.h
    fs(domain::Domain) = domain.fs
    thermal_k(domain::Domain{CType}) where CType =                              # This models α is twice the lattice α,
        CType(0.25) * (one(CType) / domain.ω_T - CType(0.5))                    # and the D3Q7 scheme uses this model α.
                                                                                # Later reconstruction of lattice parameters
                                                                                # will use α/2, e.g. for Fourier's k = α/2
    thermal_k_s(domain::Domain{CType}) where CType = CType(0.5) * domain.α_s
    thermal_k_l(domain::Domain{CType}) where CType = CType(0.5) * domain.α_l

    # Σ ϕ (T + Λ(1-fs)) on metal, plus mp T_p. TYPE_S omitted
    function enthalpy(domain::Domain{CType}) where {CType}
        flags = Array(domain.flags.data)
        TA = Array(domain.T.data)
        fsA = Array(domain.fs.data)
        Λ = domain.Λ
        @static if SURFACE
            ϕA = Array(domain.ϕ.data)
            mpA = Array(domain.mp.data)
            Tp = domain.T_p
        end
        s = 0.0
        @inbounds for n in eachindex(flags)
            fl = flags[n]
            (fl & TYPE_S) != 0x00 && continue
            @static if SURFACE
                su = fl & TYPE_SU
                if su == TYPE_F || su == TYPE_I
                    fill = ϕA[n]
                    fill < zero(CType) && (fill = zero(CType))
                    γn = blend_phase(fsA[n], domain.γ_s, domain.γ_l)
                    s += Float64(fill) * Float64(cell_enthalpy(TA[n], fsA[n], Λ, γn))
                end
                s += Float64(mpA[n]) * Float64(sensible_H(Tp, domain.γ_s))      # Enthalpy of unmelted powder
            else
                γn = blend_phase(fsA[n], domain.γ_s, domain.γ_l)
                s += Float64(cell_enthalpy(TA[n], fsA[n], Λ, γn))
            end
        end
        return CType(s)
    end

    # Instantaneous Σ Q on cells that collide T (F/I, or non-solid)
    function heat_source(domain::Domain{CType}) where {CType}
        flags = Array(domain.flags.data)
        QA = Array(domain.Q.data)
        s = 0.0
        @inbounds for n in eachindex(flags)
            fl = flags[n]
            (fl & TYPE_S) != 0x00 && continue
            @static if SURFACE
                su = fl & TYPE_SU
                (su == TYPE_F || su == TYPE_I) || continue
            end
            s += Float64(QA[n])
        end
        return CType(s)
    end

    function reset_energy_budget!(domain::Domain{CType}) where {CType}
        fill!(domain.Eacc.data, zero(CType))
        domain.E_powder = zero(CType)
        domain.H0 = enthalpy(domain)
        return domain
    end

    # H = H0 + Q - rad - evap + powder - wall + residual (streaming/BC leak)
    function energy_budget(domain::Domain{CType}) where {CType}
        acc = Array(domain.Eacc.data)
        Q = acc[EACC_Q]
        rad = acc[EACC_RAD]
        evap = acc[EACC_EVAP]
        wall = acc[EACC_WALL]
        powder = acc[EACC_POWDER] + domain.E_powder
        H = enthalpy(domain)
        expected = domain.H0 + Q - rad - evap + powder - wall
        residual = H - expected
        return (; H, H0=domain.H0, Q, rad, evap, wall, powder, expected, residual)
    end

    @static if SURFACE
        # dx is neighbor coordinate x+0 (resting), x+1 or x-1
        @inline function _wrap_mass(x, dx, N)
            ifelse(dx == 0, x,
                ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                               ifelse(x == 0, N - 1, x - 1)))
        end

        # Count the interface/fluid cells to distribute the excess mass to
        function _massex_recipients(flags, fsA, n::Int, Nx::Int, Ny::Int, Nz::Int)
            n0 = n - 1
            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)
            cnt = 0
            @inbounds for i in 2:length(VELOCITIES[:D3Q19])
                ci = VELOCITIES[:D3Q19][i]
                j = _wrap_mass(x, ci[1], Nx) +
                    _wrap_mass(y, ci[2], Ny) * Nx +
                    _wrap_mass(z, ci[3], Nz) * Nx * Ny + 1
                suj = flags[j] & (TYPE_SU | TYPE_S)
                liquid = suj == TYPE_F || suj == TYPE_I || suj == TYPE_IF || suj == TYPE_GI
                liquid = liquid && (one(eltype(fsA)) - fsA[j]) >= eltype(fsA)(1e-3)         # Count as recipient if liquid fraction (1-fs)
                                                                                            # is greater than 1e-3 (0.1%)
                cnt += Int(liquid)
            end
            return cnt
        end

        # Sum the mass on all cells (Σ mass + mp + (massex × recipients))
        function metal_mass(domain::Domain{CType}) where {CType}
            flags = Array(domain.flags.data)
            mA = Array(domain.mass.data)
            mxA = Array(domain.massex.data)
            mpA = Array(domain.mp.data)
            fsA = Array(domain.fs.data)
            Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
            s = 0.0
            @inbounds for n in eachindex(flags)
                (flags[n] & TYPE_S) != 0x00 && continue
                s += Float64(mA[n]) + Float64(mpA[n])
                mx = Float64(mxA[n])
                if mx != 0
                    cnt = _massex_recipients(flags, fsA, n, Nx, Ny, Nz)
                    s += cnt > 0 ? mx * cnt : mx
                end
            end
            return CType(s)
        end

        function reset_mass_budget!(domain::Domain{CType}) where {CType}
            fill!(domain.Macc.data, zero(CType))
            domain.M_powder = zero(CType)
            domain.M0 = metal_mass(domain)
            return domain
        end

        # M = M0 + powder − evap + residual (FSLBM/BC leak)
        function mass_budget(domain::Domain{CType}) where {CType}
            acc = Array(domain.Macc.data)
            evap = acc[MACC_EVAP]
            powder = acc[MACC_POWDER] + domain.M_powder
            M = metal_mass(domain)
            expected = domain.M0 + powder - evap
            residual = M - expected
            return (; M, M0=domain.M0, evap, powder, expected, residual)
        end
    end
end

τ(domain::Domain{CType}) where CType = CType(3) * domain.ν + one(CType) / CType(2)

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end