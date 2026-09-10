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

    @static if SURFACE
        phi::MemoryContainer{CType, Aρ}
        cached_surface_0_even!::Any
        cached_surface_0_odd!::Any
        cached_surface_1!::Any
        cached_surface_2_even!::Any
        cached_surface_2_odd!::Any
        cached_surface_3!::Any
    end

    weights::NTuple{Q, CType}
    velocities::NTuple{Q, SVector{3, Int}}

    cached_collide_even!::Any
    cached_collide_odd!::Any
    cached_initialize!::Any
    cached_moments_even!::Any
    cached_moments_odd!::Any

    initialized::Bool
end

function Model(
    Nx, Ny, Nz, ν;
    fx = 0.0f0, fy = 0.0f0, fz = 0.0f0,
    σ = 0.0f0,
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

    @static if SURFACE
        cached_surface_0_even = surface_0_even_kernel!(backend, workgroup)
        cached_surface_0_odd = surface_0_odd_kernel!(backend, workgroup)
        cached_surface_1 = surface_1_kernel!(backend, workgroup)
        cached_surface_2_even = surface_2_even_kernel!(backend, workgroup)
        cached_surface_2_odd = surface_2_odd_kernel!(backend, workgroup)
        cached_surface_3 = surface_3_kernel!(backend, workgroup)
    end

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
            CType(fx), CType(fy), CType(fz),
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

    @static if SURFACE
        buffers_ϕ = [ϕ(domains[d]) for d in 1:D]
        ϕc = attach(buffers_ϕ, Nx, Ny, Nz, Dx, Dy, Dz, "phi")
    end

    @static if SURFACE
        Model(
            scheme,
            backend, workgroup,
            Nx, Ny, Nz,
            Dx, Dy, Dz,
            domains,
            ρc, uc, fic, fc,
            # --- SURFACE --- 
            ϕc,        
            cached_surface_0_even,
            cached_surface_0_odd,
            cached_surface_1,
            cached_surface_2_even,
            cached_surface_2_odd,
            cached_surface_3,
            # --- SURFACE --- 
            w, c,
            cached_collide_even,
            cached_collide_odd,
            cached_initialize,
            cached_moments_even,
            cached_moments_odd,
            false
        )
    else
        Model(
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
            false
        )
    end
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

@static if SURFACE
    σ(model::Model) = model.σ
end

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

    pvd_path = joinpath(dir, "lbm")
    pvd = paraview_collection(pvd_path; append = isfile(pvd_path * ".pvd"))

    vtk_grid(joinpath(dir, @sprintf("lbm_%08d", t)), 0:Nx-1, 0:Ny-1, 0:Nz-1) do vtk
        vtk["rho"] = ρ3 
        vtk["u"] = (ux, uy, uz)
        vtk["flags"] = flags3
        @static if SURFACE
            vtk["phi"] = reshape(Float32.(Array(domain.ϕ.data)), Nx, Ny, Nz)
        end
        pvd[t] = vtk
    end

    vtk_save(pvd)
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
    @info "starting init"
    kernel = model.cached_initialize!
    for domain in model.domains
        N = get_N(domain)
        @static if SURFACE
            kernel(
                domain.ρ.data,
                domain.u.data,
                domain.fi.data,
                domain.flags.data,
                domain.mass.data, domain.massex.data, domain.ϕ.data,
                model.weights, model.velocities,
                Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
                ndrange = N
            )
        else
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
    end

    KernelAbstractions.synchronize(model.backend)
    model.initialized = true
    @info "finished init"
end

function step!(model::Model)
    for domain in model.domains
        N = get_N(domain)
        t_odd = isodd(domain.t)
        Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
        Nd = Int(domain.N)

        @static if SURFACE
            s0 = t_odd ? model.cached_surface_0_odd! : model.cached_surface_0_even!
            s0(domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
               domain.mass.data, domain.massex.data, domain.ϕ.data,
               model.weights, model.velocities,
               domain.fx, domain.fy, domain.fz, domain.σ,
               Nd, Nx, Ny, Nz; ndrange = N)
        end

        kernel = t_odd ? model.cached_collide_odd! : model.cached_collide_even!
        @static if SURFACE
            kernel(domain.flags.data, domain.fi.data,
                   domain.ρ.data, domain.u.data, domain.mass.data,
                   model.weights, model.velocities, domain.ω,
                   domain.fx, domain.fy, domain.fz,
                   Nd, Nx, Ny, Nz; ndrange = N)
        else
            kernel(domain.flags.data, domain.fi.data,
                   model.weights, model.velocities, domain.ω,
                   Nd, Nx, Ny, Nz; ndrange = N)
        end

        @static if SURFACE
            model.cached_surface_1!(domain.flags.data, model.velocities,
                Nd, Nx, Ny, Nz; ndrange = N)
            s2 = t_odd ? model.cached_surface_2_odd! : model.cached_surface_2_even!
            s2(domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
               model.weights, model.velocities, Nd, Nx, Ny, Nz; ndrange = N)
            model.cached_surface_3!(
                domain.ρ.data, domain.flags.data, domain.mass.data,
                domain.massex.data, domain.ϕ.data, model.velocities,
                Nd, Nx, Ny, Nz; ndrange = N)
        end

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

@static if SURFACE

@inline function surface_0_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, mass, massex, ϕ,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    flagsn = flags[n]
    bo = flagsn & TYPE_BO
    su = flagsn & TYPE_SU
    (bo == TYPE_S || su == TYPE_G) && return nothing

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)

    massn = mass[n]
    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        massn += massex[src]
    end

    NP = (Q - 1) ÷ 2
    fn1 = CType(fi[f_index(n, 1, N)])

    if su == TYPE_F
        for k in 1:NP
            i = 2k
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fp_in,  fm_in  = load_pair(fi, n, src, i, t_odd, N, CType)
            fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
            massn += (fp_in - fp_out) + (fm_in - fm_out)
        end
        mass[n] = massn
        return nothing
    end

    # TYPE_I
    ρn = fn1
    ux = zero(CType); uy = zero(CType); uz = zero(CType)
    for k in 1:NP
        i = 2k
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
        ρn += fp_out + fm_out
        ux += CType(c[i][1])*fp_out + CType(c[i+1][1])*fm_out
        uy += CType(c[i][2])*fp_out + CType(c[i+1][2])*fm_out
        uz += CType(c[i][3])*fp_out + CType(c[i+1][3])*fm_out
    end
    invρ = one(CType) / ρn
    ux *= invρ; uy *= invρ; uz *= invρ
    cs = CType(1) / sqrt(CType(3))
    ux = clamp(ux, -cs, cs); uy = clamp(uy, -cs, cs); uz = clamp(uz, -cs, cs)

    ϕin = calculate_phi(ρn, massn, flagsn)
    # σ=0 -> ρ_gas = 1; curvature later
    ρ_gas = one(CType)
    @static if VOLUME_FORCE
        uxg = clamp(ux + fx / (CType(2) * ρn), -cs, cs)
        uyg = clamp(uy + fy / (CType(2) * ρn), -cs, cs)
        uzg = clamp(uz + fz / (CType(2) * ρn), -cs, cs)
    else
        uxg, uyg, uzg = ux, uy, uz
    end
    uug = CType(1.5) * (uxg*uxg + uyg*uyg + uzg*uzg)

    for k in 1:NP
        i = 2k
        cp, cm = c[i], c[i + 1]
        srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        sup = flags[srcp] & TYPE_SU
        sum_ = flags[srcm] & TYPE_SU
        ϕp = ϕ[srcp]; ϕm = ϕ[srcm]

        fp_in,  fm_in  = load_pair(fi, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(fi, n, srcp, i, t_odd, N, CType)

        if (sup & (TYPE_F | TYPE_I)) != 0x00
            fluxp = fm_in - fp_out
            massn += sup == TYPE_F ? fluxp : CType(0.5) * (ϕp + ϕin) * fluxp
        end
        if (sum_ & (TYPE_F | TYPE_I)) != 0x00
            fluxm = fp_in - fm_out
            massn += sum_ == TYPE_F ? fluxm : CType(0.5) * (ϕm + ϕin) * fluxm
        end

        fegp = feq(w[i],     ρ_gas, uxg, uyg, uzg, uug, cp, CType)
        fegm = feq(w[i + 1], ρ_gas, uxg, uyg, uzg, uug, cm, CType)
        fp_rec = fegm - fm_out + fegp
        fm_rec = fegp - fp_out + fegm
        store_reconstructed_pair!(
            fi, n, srcp, i, fm_rec, fp_rec,
            sup == TYPE_G, sum_ == TYPE_G, t_odd, N)
    end
    mass[n] = massn
    return nothing
end

@kernel function surface_0_even_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ),
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(false), fi, ρ, u, flags, mass, massex, ϕ, w, c, fx, fy, fz, σ, N, Nx, Ny, Nz, Int(n))
end

@kernel function surface_0_odd_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ),
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(true), fi, ρ, u, flags, mass, massex, ϕ, w, c, fx, fy, fz, σ, N, Nx, Ny, Nz, Int(n))
end

end