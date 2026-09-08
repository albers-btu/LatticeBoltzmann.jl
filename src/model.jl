struct Model
    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Dx::UInt # lattice domain x
    Dy::UInt # lattice domain y
    Dz::UInt # lattice domain z

    domains::Vector{Domain}
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

    domains = Vector{Domain}(undef, D)
    
    ν = Float32(ν)
    τ = 3*ν / 1/2

    for d in 1:D
        x = UInt(d % (Dx * Dy)) % Dx
        y = UInt(d % (Dx * Dy)) ÷ Dx
        z = UInt(d % (Dx * Dy))

        nx = UInt(Nx ÷ Dx + UInt(2) * Hx)
        ny = UInt(Ny ÷ Dy + UInt(2) * Hy)
        nz = UInt(Nz ÷ Dz + UInt(2) * Hz)

        Ox = Int(x * Nx ÷ Dx) - Int(Hx)
        Oy = Int(y * Ny ÷ Dy) - Int(Hy)
        Oz = Int(z * Nz ÷ Dz) - Int(Hz)

        fx = 0.0f0
        fy = 0.0f0
        fz = 0.0f0

        σ = Float32(0.0f0)
        α = Float32(1.0f0)
        β = Float32(1.0f0)

        domains[d] = Domain(
            nx, ny, nz,
            Ox, Oy, Oz,
            ν, 
            fx, fy, fz,
        )
    end

    return Model(
        Nx, Ny, Nz,
        Dx, Dy, Dz,
        domains
    )
end

# complete model computation size
function get_N(model::Model)
    model.Nx * model.Ny * model.Nz
end

# size of domains inside model
function get_D(model::Model)
    model.Dx * model.Dy * model.Dz
end

function initialize(model::Model)
    @warn "todo"
end

function run(model::Model, steps::Int)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    for i in 1:steps
        @info "step $i"
    end
end