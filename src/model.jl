using Printf

mutable struct Model{
    Aρ<:AbstractArray{Float32},
    Au<:AbstractArray{Float32},
    Afi<:AbstractArray{Float32},
    Af<:AbstractArray{UInt8},
    Q
}
    scheme::Symbol

    backend::KernelAbstractions.Backend
    workgroup::Int

    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Dx::UInt # lattice domain x
    Dy::UInt # lattice domain y
    Dz::UInt # lattice domain z

    domains::Vector{Domain{Aρ, Au, Afi, Af}}

    ρ::MemoryContainer{Float32, Aρ}
    u::MemoryContainer{Float32, Au}
    fi::MemoryContainer{Float32, Afi}
    flags::MemoryContainer{UInt8, Af}

    weights::NTuple{Q, Float32}
    velocities::NTuple{Q, SVector{3, Int}}

    cached_collide!::Any # cached kernel
    cached_initialize!::Any # cached kernel

    initialized::Bool
end

function Model(Nx, Ny, Nz, ν; scheme = :D3Q19, backend = CPU(), workgroup = default_workgroup(backend))
    backend isa CUDABackend && !CUDA.functional() && throw(ArgumentError("CUDABackend requested but CUDA is not functional"))

    w = weights(scheme)
    c = velocities(scheme)
    Q = length(w)

    cached_collide = stream_collide_kernel!(backend, workgroup)
    cached_initialize = initialize_kernel!(backend, workgroup)

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

    domains = map(1:Int(D)) do d
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

        Domain(
            nx, ny, nz,
            Ox, Oy, Oz,
            ν,
            0.0f0, 0.0f0, 0.0f0,
            scheme,
            backend
        )
    end

    buffers_ρ = [ρ(domains[d]) for d in 1:D]
    buffers_u = [u(domains[d]) for d in 1:D]
    buffers_fi = [fi(domains[d]) for d in 1:D]
    buffers_flags = [flags(domains[d]) for d in 1:D]

    ρc = attach(buffers_ρ, Nx, Ny, Nz, Dx, Dy, Dz, "rho")
    uc = attach(buffers_u, Nx, Ny, Nz, Dx, Dy, Dz, "u")
    fic = attach(buffers_fi, Nx, Ny, Nz, Dx, Dy, Dz, "fi")
    fc = attach(buffers_flags, Nx, Ny, Nz, Dx, Dy, Dz, "flags")

    return Model(
        scheme,
        backend, workgroup,
        Nx, Ny, Nz,
        Dx, Dy, Dz,
        domains,
        ρc, uc, fic, fc,
        w, c,
        cached_collide, cached_initialize,
        false
    )
end

arraytype(::CPU) = Array
arraytype(::CUDABackend) = CuArray

default_workgroup(::CPU) = 64
default_workgroup(::CUDABackend) = 256

get_N(model::Model)= Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
get_D(model::Model) = Int(model.Dx) * Int(model.Dy) * Int(model.Dz)

flags(model::Model) = model.flags

ρ(model::Model) = model.ρ
u(model::Model) = model.u

# sets up the simulation, copies data to device and runs for n steps
function run!(model::Model, steps::Int)
    steps > 0 || throw(ArgumentError("steps must be positive"))

    if !model.initialized
        initialize!(model)
    end

    for i in 1:steps
        start = time_ns()
        step!(model)
        elapsed_s = (time_ns() - start) / 1e9
        mlups = get_N(model)*1e-6 / elapsed_s
        @info @sprintf("%.2f MLUPS", mlups)
    end
end

function initialize!(model::Model)
    kernel = model.cached_initialize!
    for domain in model.domains
        N = get_N(domain)
        kernel(
            domain.ρ.data,
            domain.u.data,
            domain.fi.data,
            domain.flags.data,
            model.weights, model.velocities,
            N;
            ndrange = N
        )
    end

    KernelAbstractions.synchronize(model.backend)
    model.initialized = true
    @info "finished initializing"
end

function step!(model::Model)
    kernel = model.cached_collide!
    for (d, domain) in enumerate(model.domains)
        N = get_N(domain)
        kernel(
            domain.flags.data,
            domain.fi.data,
            domain.fo.data,
            model.weights, model.velocities,
            1.0f0 / τ(domain),
            Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
            ndrange = N
        )
        domain.fi, domain.fo = domain.fo, domain.fi
        model.fi.buffers[d] = domain.fi
        increment_time_step!(domain, 1)
    end

    KernelAbstractions.synchronize(model.backend)
end