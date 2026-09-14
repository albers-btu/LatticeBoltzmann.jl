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
    end

    @static if TEMPERATURE
        α::CType
        α_s::CType            # solid Model-α (twice CE diffusivity)
        α_l::CType            # liquid Model-α
        ν_s::CType
        ν_l::CType
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
    ν_s::CType = zero(CType),
    ν_l::CType = zero(CType),
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
                ϕ, mass, massex,
                αT, αs, αl, νs, νl, β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
                Λ_v, T_v, C_hk, p0v, β_v,
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
                ϕ, mass, massex,
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
                αT, αs, αl, νs, νl, β, T_avg, ω_T, Tmem, gi, Qmem, hmem, Λ, Ts, Tl, K0, fsmem,
                Λ_v, T_v, C_hk, p0v, β_v,
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
end

τ(domain::Domain{CType}) where CType = CType(3) * domain.ν + CType(1) / CType(2)

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end