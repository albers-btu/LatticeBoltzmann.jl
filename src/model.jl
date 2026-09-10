using Printf, CUDA, WriteVTK

mutable struct Model{
    CType<:AbstractFloat,
    SType<:AbstractFloat,
    Aρ<:AbstractArray{CType},
    Au<:AbstractArray{CType},
    Afi<:AbstractArray{SType},
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

    domains::Vector{<:Domain{CType, SType}}

    ρ::MemoryContainer{CType, Aρ}
    u::MemoryContainer{CType, Au}
    fi::MemoryContainer{SType, Afi}
    flags::MemoryContainer{UInt8, Af}

    weights::NTuple{Q, CType}
    velocities::NTuple{Q, SVector{3, Int}}

    cached_collide_even!::Any
    cached_collide_odd!::Any
    cached_initialize!::Any
    cached_moments_even!::Any
    cached_moments_odd!::Any

    initialized::Bool

    pvd::Any
end

function Model(
    Nx, Ny, Nz, ν; 
    CType::Type{<:AbstractFloat} = Float32,
    SType::Type{<:AbstractFloat} = CType,
    scheme = :D3Q19, 
    backend = CPU(), 
    workgroup = default_workgroup(backend)
)
    backend isa CUDABackend && !CUDA.functional() && throw(ArgumentError("CUDABackend requested but CUDA is not functional"))

    w = weights(scheme, CType)
    c = velocities(scheme)

    cached_collide_even = stream_collide_even_kernel!(backend, workgroup)
    cached_collide_odd = stream_collide_odd_kernel!(backend, workgroup)
    cached_initialize = initialize_kernel!(backend, workgroup)
    cached_moments_even = moments_even_kernel!(backend, workgroup)
    cached_moments_odd = moments_odd_kernel!(backend, workgroup)

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
    
    ν = CType(ν)

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
            backend,
            CType,
            SType
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
        cached_collide_even,
        cached_collide_odd,
        cached_initialize,
        cached_moments_even,
        cached_moments_odd,
        false,
        nothing
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

function export!(model::Model; dir::AbstractString="output")
    model.initialized || initialize!(model)
    moments!(model)

    mkpath(dir)

    domain = model.domains[1] # not general yet
    t = Int(domain.t)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)

    ρ_host     = Array(domain.ρ.data)
    u_host     = Array(domain.u.data)
    flags_host = Array(domain.flags.data)

    ρ3     = reshape(Float32.(ρ_host), Nx, Ny, Nz)
    ux     = reshape(Float32.(view(u_host, :, 1)), Nx, Ny, Nz)
    uy     = reshape(Float32.(view(u_host, :, 2)), Nx, Ny, Nz)
    uz     = reshape(Float32.(view(u_host, :, 3)), Nx, Ny, Nz)
    flags3 = reshape(flags_host, Nx, Ny, Nz)

    if model.pvd === nothing
        model.pvd = paraview_collection(joinpath(dir, "lbm"))
    end

    vtk_grid(joinpath(dir, @sprintf("lbm_%08d", t)), 0:Nx-1, 0:Ny-1, 0:Nz-1) do vtk
        vtk["rho"] = ρ3 
        vtk["u"] = (ux, uy, uz)
        vtk["flags"] = flags3
        model.pvd[t] = vtk
    end

    vtk_save(model.pvd)
    return nothing
end

function run!(model::Model, nsteps::Int)
    nsteps > 0 || throw(ArgumentError("nsteps must be positive"))

    if !model.initialized
        initialize!(model)
    end

    for _ in 1:nsteps
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
            Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
            ndrange = N
        )
    end

    KernelAbstractions.synchronize(model.backend)
    model.initialized = true
    @info "finished initializing"
end

function step!(model::Model)
    for domain in model.domains
        N = get_N(domain)
        kernel = isodd(domain.t) ? model.cached_collide_odd! : model.cached_collide_even!
        kernel(
            domain.flags.data,
            domain.fi.data,
            model.weights, model.velocities,
            domain.ω,
            Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
            ndrange = N
        )
        increment_time_step!(domain, 1)
    end
    KernelAbstractions.synchronize(model.backend)
end

function moments!(model::Model)
    for domain in model.domains
        N = get_N(domain)
        kernel = isodd(domain.t) ? model.cached_moments_odd! : model.cached_moments_even!
        kernel(
            domain.ρ.data,
            domain.u.data,
            domain.flags.data,
            domain.fi.data,
            model.weights, model.velocities,
            Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
            ndrange = N
        )
    end
    KernelAbstractions.synchronize(model.backend)
end