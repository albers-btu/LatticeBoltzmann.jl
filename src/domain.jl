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

    ρ::Memory{CType, Aρ}
    u::Memory{CType, Au}
    fi::Memory{SType, Afi}
    flags::Memory{UInt8, Af}

    @static if SURFACE
        ϕ::Memory{CType, Aρ}
        mass::Memory{CType, Aρ}
        massex::Memory{CType, Aρ}
    end

    t::UInt64
end

function Domain(Nx, Ny, Nz, Ox, Oy, Oz, ν, fx, fy, fz, scheme, backend, ::Type{CType}, ::Type{SType}; σ::CType = zero(CType)) where {CType, SType}
    Q = length(WEIGHTS[scheme])
    
    N = Int(Nx) * Int(Ny) * Int(Nz)
    ω = one(CType) / (CType(3) * CType(ν) + CType(1) / CType(2))
    AT = arraytype(backend)

    ρ = Memory(AT{CType}(undef, N))
    fill!(ρ.data, one(CType))

    u = Memory(AT{CType}(undef, N, 3))
    fill!(u.data, zero(CType))

    fi = Memory(AT{SType}(undef, N * Q))
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

    @static if SURFACE
        Domain(
            UInt(Nx), UInt(Ny), UInt(Nz),
            Int(Ox), Int(Oy), Int(Oz),
            CType(ν), N, ω,
            CType(fx), CType(fy), CType(fz),
            CType(σ),
            ρ, u, fi, flags,
            # --- SURFACE ---
            ϕ, mass, massex,
            # --- SURFACE ---
            UInt64(0)
        )
    else
        Domain(
            UInt(Nx), UInt(Ny), UInt(Nz),
            Int(Ox), Int(Oy), Int(Oz),
            CType(ν), N, ω,
            CType(fx), CType(fy), CType(fz),
            CType(σ),
            ρ, u, fi, flags,
            UInt64(0)
        )
    end
end

get_N(domain::Domain) = Int(domain.Nx) * Int(domain.Ny) * Int(domain.Nz)

ρ(domain::Domain) = domain.ρ
u(domain::Domain) = domain.u
fi(domain::Domain) = domain.fi
flags(domain::Domain) = domain.flags

@static if SURFACE
    ϕ(domain::Domain) = domain.ϕ
    mass(domain::Domain) = domain.mass
    massex(domain::Domain) = domain.massex
end

τ(domain::Domain{CType}) where CType = CType(3) * domain.ν + CType(1) / CType(2)

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end