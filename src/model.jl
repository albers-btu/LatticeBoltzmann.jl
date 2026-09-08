struct Model{Aρ<:AbstractArray{Float32}, Au<:AbstractArray{Float32}}
    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Dx::UInt # lattice domain x
    Dy::UInt # lattice domain y
    Dz::UInt # lattice domain z

    domains::Vector{Domain{Vector{Float32}, Matrix{Float32}}}

    ρ::MemoryContainer{Float32, Aρ}
    u::MemoryContainer{Float32, Au}
end

function Model(Nx, Ny, Nz, ν)
    Dx = UInt(1)
    Dy = UInt(1)
    Dz = UInt(1)
    D = UInt(Dx*Dy*Dz)

    if Nx % Dx != 0 || Ny % Dy != 0 || Nz % Dz != 0
        @warn "grid is not equally divisible in domains"
    end

    Nx::UInt = UInt(Nx)
    Ny::UInt = UInt(Ny)
    Nz::UInt = UInt(Nz)

    Hx::UInt = UInt(Dx > 1) # halo offset x
    Hy::UInt = UInt(Dy > 1) # halo offset y
    Hz::UInt = UInt(Dz > 1) # halo offset z
    
    ν = Float32(ν)
    τ = 3*ν + 1/2

    domains = Vector{Domain{Vector{Float32}, Matrix{Float32}}}(undef, D)
    for d in 1:D
        d0 = d - 1
        x = UInt(d0 % (Dx * Dy)) % Dx
        y = UInt(d0 % (Dx * Dy)) ÷ Dx
        z = UInt(d0 ÷ (Dx * Dy))

        nx = Nx ÷ Dx + 2 * Hx
        ny = Ny ÷ Dy + 2 * Hy
        nz = Nz ÷ Dz + 2 * Hz

        Ox = Int(x * Nx ÷ Dx) - Int(Hx)
        Oy = Int(y * Ny ÷ Dy) - Int(Hy)
        Oz = Int(z * Nz ÷ Dz) - Int(Hz)

        domains[d] = Domain(
            nx, ny, nz,
            Ox, Oy, Oz,
            ν,
            0.0f0, 0.0f0, 0.0f0
        )
    end

    buffers_ρ = [ρ(domains[d]) for d in 1:D]
    buffers_u = [u(domains[d]) for d in 1:D]

    ρc = attach(buffers_ρ, Nx, Ny, Nz, Dx, Dy, Dz, "rho")
    uc = attach(buffers_u, Nx, Ny, Nz, Dx, Dy, Dz, "u")

    return Model(
        Nx, Ny, Nz,
        Dx, Dy, Dz,
        domains,
        ρc, uc
    )
end

get_N(model::Model)= Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
get_D(model::Model) = Int(model.Dx) * Int(model.Dy) * Int(model.Dz)

ρ(model::Model) = model.ρ
u(model::Model) = model.u

function initialize(model::Model)
    @warn "todo"
end

function run(model::Model, steps::Int)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    for i in 1:steps
        @info "step $i"
    end
end