mutable struct Domain{
    DType<:AbstractFloat,
    Aρ<:AbstractArray{DType}, 
    Au<:AbstractArray{DType},
    Afi<:AbstractArray{DType},
    Af<:AbstractArray{UInt8}
}
    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Ox::Int # offset x
    Oy::Int # offset y
    Oz::Int # offset z

    ν::DType # kinematic shear viscosity
    N::Int
    ω::DType

    fx::DType # global force per volume x
    fy::DType # global force per volume y
    fz::DType # global force per volume z

    ρ::Memory{DType, Aρ}
    u::Memory{DType, Au}
    fi::Memory{DType, Afi}
    flags::Memory{UInt8, Af}

    t::UInt64
end

function Domain(Nx, Ny, Nz, Ox, Oy, Oz, ν, fx, fy, fz, scheme, backend, ::Type{DType}) where {DType}
    Q = length(WEIGHTS[scheme])
    
    N = Int(Nx) * Int(Ny) * Int(Nz)
    ω = one(DType) / (DType(3) * DType(ν) + DType(1) / DType(2))
    AT = arraytype(backend)

    ρ = Memory(AT{DType}(undef, N))
    fill!(ρ.data, one(DType))

    u = Memory(AT{DType}(undef, N, 3))
    fill!(u.data, zero(DType))

    fi = Memory(AT{DType}(undef, N * Q))
    fill!(fi.data, zero(DType))

    flags = Memory(AT{UInt8}(undef, N))
    fill!(flags.data, 0x00)

    Domain(
        UInt(Nx), UInt(Ny), UInt(Nz),
        Int(Ox), Int(Oy), Int(Oz),
        DType(ν), N, ω,
        DType(fx), DType(fy), DType(fz),
        ρ, u, fi, flags,
        UInt64(0)
    )
end

get_N(domain::Domain) = Int(domain.Nx) * Int(domain.Ny) * Int(domain.Nz)

ρ(domain::Domain) = domain.ρ
u(domain::Domain) = domain.u
fi(domain::Domain) = domain.fi
flags(domain::Domain) = domain.flags

τ(domain::Domain{DType}) where DType = DType(3) * domain.ν + DType(1) / DType(2)

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end