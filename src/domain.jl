struct Domain{Aρ<:AbstractArray{Float32}, Au<:AbstractArray{Float32}}
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
end

function Domain(Nx, Ny, Nz, Ox, Oy, Oz, ν, fx, fy, fz)
    N = Int(Nx) * Int(Ny) * Int(Nz)
    Domain(
        UInt(Nx), UInt(Ny), UInt(Nz),
        Int(Ox), Int(Oy), Int(Oz),
        Float32(ν), Float32(fx), Float32(fy), Float32(fz),
        Memory(fill(1.0f0, N)),
        Memory(zeros(Float32, N, 3))
    )
end

get_N(domain::Domain) = Int(domain.Nx) * Int(domain.Ny) * Int(domain.Nz)

ρ(domain::Domain) = domain.ρ
u(domain::Domain) = domain.u