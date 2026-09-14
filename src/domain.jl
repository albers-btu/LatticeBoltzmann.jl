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
    end

    @static if TEMPERATURE
        α::CType
        α_s::CType            # solid Model-α at T_avg (twice CE diffusivity)
        α_l::CType
        α_sT::CType           # dα_s / dT_lat
        α_lT::CType
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
                αT, αs, αl, CType(α_sT), CType(α_lT), νs, νl, CType(ν_sT), CType(ν_lT), β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
                Λ_v, T_v, C_hk, p0v, β_v, CType(C_rad), CType(T_rad),
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
                αT, αs, αl, CType(α_sT), CType(α_lT), νs, νl, CType(ν_sT), CType(ν_lT), β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
                Λ_v, T_v, C_hk, p0v, β_v, CType(C_rad), CType(T_rad),
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
                    s += Float64(fill) * Float64(cell_enthalpy(TA[n], fsA[n], Λ))
                end
                s += Float64(mpA[n]) * Float64(Tp)
            else
                s += Float64(cell_enthalpy(TA[n], fsA[n], Λ))
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
end

τ(domain::Domain{CType}) where CType = CType(3) * domain.ν + CType(1) / CType(2)

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end