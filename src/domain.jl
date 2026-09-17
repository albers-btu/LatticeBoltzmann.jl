# Cumulative lattice-enthalpy account (into-metal Q, out-of-metal rad/evap/wall).
const EACC_Q = 1
const EACC_RAD = 2
const EACC_EVAP = 3
const EACC_WALL = 4
const EACC_POWDER = 5
const EACC_N = 5

# Metal-mass account matching energy: M = M0 + powder − evap + residual.
const MACC_EVAP = 1
const MACC_POWDER = 2
const MACC_N = 2

mutable struct Domain{
    CType<:AbstractFloat,
    SType<:AbstractFloat,
    Aρ<:AbstractArray{CType}, 
    Au<:AbstractArray{CType},
    Afi<:AbstractArray{SType},
    Af<:AbstractArray{UInt8}
}
    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Ox::Int # offset x
    Oy::Int # offset y
    Oz::Int # offset z

    ν::CType # kinematic shear viscosity
    N::Int
    ω::CType

    fx::CType # global force per volume x
    fy::CType # global force per volume y
    fz::CType # global force per volume z

    σ::CType
    σT::CType  # dσ/dT; 0 → constant σ
    Tσ::CType  # T_ref in σ(T) = σ + σT (T - Tσ)

    ρ::Memory{CType, Aρ}
    u::Memory{CType, Au}
    F::Memory{CType, Au}
    fi::Memory{SType, Afi}
    flags::Memory{UInt8, Af}

    @static if SURFACE
        ϕ::Memory{CType, Aρ}
        mass::Memory{CType, Aρ}
        massex::Memory{CType, Aρ}
        msrc::Memory{CType, Aρ}  # feed Δmass/(ρ Δt); 0 → none
        mp::Memory{CType, Aρ}    # unmelted powder mass (same units as mass)
        τ_p::CType               # powder lifetime (lattice steps); 0 → msrc→mass
        T_p::CType               # powder temperature (lattice)
        p_gas::Memory{CType, Aρ} # lattice gas pressure for reconstruction; p_atm=1/3
        c::Memory{CType, Aρ}     # dissolved gas (D3Q7; c = Σg)
        ci::Memory{SType, Afi}   # D3Q7 populations for c
        nflux::Memory{CType, Aρ} # Henry Δn this step (n-units)
        bid::Memory{CType, Aρ}   # enclosed bubble id on G/I; 0 = none
        ω_c::CType               # D3Q7 ω for dissolved; 0 → off
        k_H::CType               # Henry c = k_H p; 0 → no interface exchange
        a::Memory{CType, Aρ}     # blowing-agent amount (n-units)
        a_res::Memory{CType, Aρ} # decomposed residue (same units)
        k_a::CType               # Arrhenius k0 [1/step]; 0 → off
        E_a::CType               # activation in lattice T (E/K)
        Y_a::CType               # yield: Δn_dissolved = Y_a (−Δa)
        a_fs_max::CType          # decompose only if fs < this; 1 → solid too
    end

    @static if TEMPERATURE
        α::CType
        α_s::CType            # solid Model-α at T_avg (twice CE diffusivity)
        α_l::CType
        α_sT::CType           # dα_s / dT_lat
        α_lT::CType
        γ_s::CType            # d(cp/cp_ref)/dT_lat; 0 → constant cp
        γ_l::CType
        ν_s::CType
        ν_l::CType
        ν_sT::CType           # dν_s / dT_lat
        ν_lT::CType
        β::CType
        T_avg::CType
        ω_T::CType
        T::Memory{CType, Aρ}
        gi::Memory{SType, Afi}
        Q::Memory{CType, Aρ}
        h::Memory{CType, Aρ}  # Robin h; 0 -> pure Neumann
        Λ::CType              # latent L/cp in lattice T; 0 → no melting
        Ts::CType             # solidus
        Tl::CType             # liquidus (Ts=Tl → isothermal Stefan)
        K0::CType             # Kozeny–Carman K0; 0 → no Darcy
        fs::Memory{CType, Aρ} # solid fraction
        Λ_v::CType            # vaporization L_v/(cp K); 0 → no evaporation
        T_v::CType            # boiling T (lattice)
        C_hk::CType           # Hertz–Knudsen prefactor (lattice)
        p0v::CType            # p_atm (lattice)
        β_v::CType            # L_v/(R_sp K) Clausius–Clapeyron
        C_rad::CType          # εσ K³ s/(ρ cp m); Q = C_rad (T^4-T_∞^4); 0 → off
        T_rad::CType          # far-field T for radiation (lattice)
        Eacc::Memory{CType, Aρ}
        H0::CType
        E_powder::CType       # host: jet deposit × T_p
        @static if SURFACE
            Macc::Memory{CType, Aρ}
            M0::CType
            M_powder::CType   # host: jet deposit mass
        end
    end

    t::UInt64
end

function Domain(Nx, Ny, Nz, Ox, Oy, Oz, ν, fx, fy, fz, scheme, backend, ::Type{CType}, ::Type{SType};
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
    α_c::CType = zero(CType),
    k_H::CType = zero(CType),
    k_a::CType = zero(CType),
    E_a::CType = zero(CType),
    Y_a::CType = one(CType),
    a_fs_max::CType = one(CType),
) where {CType, SType}
    nvel = length(WEIGHTS[scheme])

    N = Int(Nx) * Int(Ny) * Int(Nz)
    ω = one(CType) / (CType(3) * CType(ν) + CType(1) / CType(2))
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

        p_gas = Memory(AT{CType}(undef, N))
        fill!(p_gas.data, CType(1) / CType(3))

        cmem = Memory(AT{CType}(undef, N))
        fill!(cmem.data, one(CType))
        cimem = Memory(AT{SType}(undef, N * 7))
        fill!(cimem.data, zero(SType))
        nflux = Memory(AT{CType}(undef, N))
        fill!(nflux.data, zero(CType))
        bid = Memory(AT{CType}(undef, N))
        fill!(bid.data, zero(CType))
        ω_c = α_c > zero(CType) ?
            one(CType) / (CType(2) * α_c + CType(1) / CType(2)) : zero(CType)
        kH = CType(k_H)
        amem = Memory(AT{CType}(undef, N))
        fill!(amem.data, zero(CType))
        ares = Memory(AT{CType}(undef, N))
        fill!(ares.data, zero(CType))
        ka = CType(k_a)
        Ea = CType(E_a)
        Ya = CType(Y_a)
        afs = CType(a_fs_max)
    end

    @static if TEMPERATURE
        αT = α == zero(CType) ? CType(ν) : α
        αs = α_s == zero(CType) ? αT : α_s
        αl = α_l == zero(CType) ? αT : α_l
        νs = ν_s == zero(CType) ? CType(ν) : ν_s
        νl = ν_l == zero(CType) ? CType(ν) : ν_l
        ω_T = one(CType) / (CType(2) * αT + CType(1) / CType(2))
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
                ϕ, mass, massex, msrc, mp, CType(τ_p), CType(T_p), p_gas,
                cmem, cimem, nflux, bid, ω_c, kH, amem, ares, ka, Ea, Ya, afs,
                αT, αs, αl, CType(α_sT), CType(α_lT), CType(γ_s), CType(γ_l), νs, νl, CType(ν_sT), CType(ν_lT), β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
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
                CType(τ_p), CType(T_p), p_gas,
                cmem, cimem, nflux, bid, ω_c, kH, amem, ares, ka, Ea, Ya, afs,
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
                αT, αs, αl, CType(α_sT), CType(α_lT), CType(γ_s), CType(γ_l), νs, νl, CType(ν_sT), CType(ν_lT), β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
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
    p_gas(domain::Domain) = domain.p_gas
    c(domain::Domain) = domain.c
    dissolved_D(domain::Domain{CType}) where CType =
        domain.ω_c > zero(CType) ?
            CType(0.25) * (one(CType) / domain.ω_c - CType(0.5)) : zero(CType)
    agent(domain::Domain) = domain.a
    agent_res(domain::Domain) = domain.a_res
    arrhenius_k(k_a, E_a, T) = k_a * exp(-E_a / max(T, eps(typeof(T))))
    function agent_inventory(domain::Domain{CType}) where {CType}
        flags = Array(domain.flags.data)
        aA = Array(domain.a.data)
        rA = Array(domain.a_res.data)
        cA = Array(domain.c.data)
        ϕA = Array(domain.ϕ.data)
        kH = domain.k_H
        Sa = 0.0
        Sr = 0.0
        Sn = 0.0
        @inbounds for n in eachindex(flags)
            su = flags[n] & TYPE_SU
            (su == TYPE_F || su == TYPE_I) || continue
            Sa += Float64(aA[n])
            Sr += Float64(rA[n])
            ϕn = Float64(ϕA[n])
            ϕn = ϕn < 0 ? 0.0 : (ϕn > 1 ? 1.0 : ϕn)
            if kH > 0
                Sn += ϕn * (Float64(cA[n]) - 1) / Float64(kH)
            end
        end
        return (; a=CType(Sa), res=CType(Sr), dissolved=CType(Sn), total=CType(Sa + Sr))
    end
end

@static if TEMPERATURE
    T(domain::Domain) = domain.T
    gi(domain::Domain) = domain.gi
    Q(domain::Domain) = domain.Q
    htc(domain::Domain) = domain.h
    fs(domain::Domain) = domain.fs
    # D3Q7 Peng c_sT² = 1/4. Model `α` sets ω_T = 1/(2α+1/2); Fourier k = α/2.
    thermal_k(domain::Domain{CType}) where CType =
        CType(0.25) * (one(CType) / domain.ω_T - CType(0.5))
    thermal_k_s(domain::Domain{CType}) where CType = CType(0.5) * domain.α_s
    thermal_k_l(domain::Domain{CType}) where CType = CType(0.5) * domain.α_l

    # Σ ϕ (T + Λ(1-fs)) on metal, plus mp T_p. TYPE_S omitted.
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
                s += Float64(mpA[n]) * Float64(sensible_H(Tp, domain.γ_s))
            else
                γn = blend_phase(fsA[n], domain.γ_s, domain.γ_l)
                s += Float64(cell_enthalpy(TA[n], fsA[n], Λ, γn))
            end
        end
        return CType(s)
    end

    # Instantaneous Σ Q on cells that collide T (F/I, or non-solid).
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

    # H = H0 + Q - rad - evap + powder - wall + residual (streaming/BC leak).
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
        # surface_3 stores massex as excess / N_liquid_neighbors.
        @inline function _wrap_mass(x, dx, N)
            ifelse(dx == 0, x,
                ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                               ifelse(x == 0, N - 1, x - 1)))
        end

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
                liquid = liquid && (one(eltype(fsA)) - fsA[j]) >= eltype(fsA)(1e-3)
                cnt += Int(liquid)
            end
            return cnt
        end

        # Σ mass + mp + massex×recipients on non-solid cells.
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

        # M = M0 + powder − evap + residual (FSLBM/BC leak).
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

τ(domain::Domain{CType}) where CType = CType(3) * domain.ν + CType(1) / CType(2)

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end