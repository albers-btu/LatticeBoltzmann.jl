mutable struct Domain{
    Aρ<:AbstractArray{Float32}, 
    Au<:AbstractArray{Float32},
    Afi<:AbstractArray{Float32},
    Af<:AbstractArray{UInt8}
}
    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Ox::Int # offset x
    Oy::Int # offset y
    Oz::Int # offset z

    ν::Float32 # kinematic shear viscosity

    fx::Float32 # global force per volume x
    fy::Float32 # global force per volume y
    fz::Float32 # global force per volume z

    ρ::Memory{Float32, Aρ}
    u::Memory{Float32, Au}
    fi::Memory{Float32, Afi}
    fo::Memory{Float32, Afi}
    flags::Memory{UInt8, Af}

    t::UInt64
end

function Domain(Nx, Ny, Nz, Ox, Oy, Oz, ν, fx, fy, fz, scheme, backend)
    Q = length(WEIGHTS[scheme])
    
    N = Int(Nx) * Int(Ny) * Int(Nz)
    AT = arraytype(backend)

    ρ = Memory(AT{Float32}(undef, N))
    fill!(ρ.data, 1.0f0)

    u = Memory(AT{Float32}(undef, N, 3))
    fill!(u.data, 0.0f0)

    fi = Memory(AT{Float32}(undef, N * Q))
    fill!(fi.data, 0.0f0)
    fo = Memory(AT{Float32}(undef, N * Q))
    fill!(fo.data, 0.0f0)

    flags = Memory(AT{UInt8}(undef, N))
    fill!(flags.data, 0x00)

    Domain(
        UInt(Nx), UInt(Ny), UInt(Nz),
        Int(Ox), Int(Oy), Int(Oz),
        Float32(ν), Float32(fx), Float32(fy), Float32(fz),
        ρ, u, fi, fo, flags,
        UInt64(0)
    )
end

get_N(domain::Domain) = Int(domain.Nx) * Int(domain.Ny) * Int(domain.Nz)

ρ(domain::Domain) = domain.ρ
u(domain::Domain) = domain.u
fi(domain::Domain) = domain.fi
flags(domain::Domain) = domain.flags

τ(domain::Domain) = 3 * domain.ν + 0.5f0

function increment_time_step!(domain::Domain, steps::Int)
    domain.t += steps
end