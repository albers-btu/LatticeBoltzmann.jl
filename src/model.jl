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
    F::MemoryContainer{CType, Au}
    fi::MemoryContainer{SType, Afi}
    flags::MemoryContainer{UInt8, Af}

    @static if TEMPERATURE
        T::MemoryContainer{CType, Aρ}
        Q::MemoryContainer{CType, Aρ}
        h::MemoryContainer{CType, Aρ}
        fs::MemoryContainer{CType, Aρ}
    end

    @static if SURFACE
        phi::MemoryContainer{CType, Aρ}
        msrc::MemoryContainer{CType, Aρ}
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
    cached_moving!::Any
    cached_update_force_even!::Any
    cached_update_force_odd!::Any
    cached_reset_force!::Any

    initialized::Bool
    units::Units{CType}
end

function Model(
    Nx, Ny, Nz, units::Units{CType};
    ν = 1.0e-6,
    gx = 0.0f0, gy = 0.0f0, gz = 0.0f0,
    σ = 0.0f0,
    σT = 0.0f0,
    Tσ = nothing,
    α = 0.0f0,
    α_s = nothing,
    α_l = nothing,
    ν_s = nothing,
    ν_l = nothing,
    β = 0.0f0,
    T_avg = 1.0f0,
    latent = 0.0f0,
    Ts = nothing,
    Tl = nothing,
    K0 = 0.0f0,
    latent_v = 0.0f0,
    T_v = nothing,
    M = 0.0558,
    p_atm = 101325.0,
    SType::Type{<:AbstractFloat} = CType,
    scheme = :D3Q19,
    backend = CPU(),
    workgroup = default_workgroup(backend)
) where {CType}
    ν  = CType(lbm_ν(units, ν))
    fx = CType(lbm_g(units, gx))
    fy = CType(lbm_g(units, gy))
    fz = CType(lbm_g(units, gz))
    σ  = CType(lbm_σ(units, σ))
    σT = CType(lbm_σT(units, σT))
    Tσl = Tσ === nothing ? CType(T_avg) :
          Tσ isa Quantity ? CType(lbm_T(units, Tσ)) : CType(Tσ)
    α  = CType(lbm_ν(units, α))
    αs = α_s === nothing ? α : CType(lbm_ν(units, α_s))
    αl = α_l === nothing ? α : CType(lbm_ν(units, α_l))
    νs = ν_s === nothing ? ν : CType(lbm_ν(units, ν_s))
    νl = ν_l === nothing ? ν : CType(lbm_ν(units, ν_l))
    Λ  = CType(lbm_Λ(units, latent))
    Tsl = Ts === nothing ? CType(T_avg) :
          Ts isa Quantity ? CType(lbm_T(units, Ts)) : CType(Ts)
    Tll = Tl === nothing ? Tsl :
          Tl isa Quantity ? CType(lbm_T(units, Tl)) : CType(Tl)
    K0l = K0 isa Quantity ? CType(ustrip(u"m^2", K0) / units.m^2) : CType(K0 / units.m^2)
    Tvl = T_v === nothing ? zero(CType) :
          T_v isa Quantity ? CType(lbm_T(units, T_v)) : CType(T_v)
    Λv, βv, p0l, Chk = lbm_evap(units, latent_v, M, p_atm)
    Lvsi = latent_v isa Quantity ? ustrip(u"J/kg", latent_v) : Float64(latent_v)
    if Lvsi > 0 && Tvl == 0
        @warn "latent_v > 0 but T_v is unset; evaporation will stay off"
        Λv = zero(CType)
    end

    # @info units

    model = Model(Nx, Ny, Nz, ν; fx, fy, fz, σ=σ, σT=σT, Tσ=Tσl, α=α, α_s=αs, α_l=αl,
                  ν_s=νs, ν_l=νl, β=CType(β), T_avg=CType(T_avg),
                  Λ=Λ, Ts=Tsl, Tl=Tll, K0=K0l,
                  Λ_v=CType(Λv), T_v=Tvl, C_hk=CType(Chk), p0v=CType(p0l), β_v=CType(βv),
                  CType, SType, scheme, backend, workgroup)
    model.units = units
    return model
end

function Model(
    Nx, Ny, Nz, ν;
    fx = 0.0f0, fy = 0.0f0, fz = 0.0f0,
    σ = 0.0f0,
    σT = 0.0f0,
    Tσ = nothing,
    α = 0.0f0,
    α_s = 0.0f0,
    α_l = 0.0f0,
    ν_s = 0.0f0,
    ν_l = 0.0f0,
    β = 0.0f0,
    T_avg = 1.0f0,
    Λ = 0.0f0,
    Ts = nothing,
    Tl = nothing,
    K0 = 0.0f0,
    Λ_v = 0.0f0,
    T_v = 0.0f0,
    C_hk = 0.0f0,
    p0v = 0.0f0,
    β_v = 0.0f0,
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
    @static if MOVING_BOUNDARIES
        cached_moving = update_moving_boundaries_kernel!(backend, workgroup)
    else
        cached_moving = nothing
    end
    @static if FORCE_FIELD
        cached_update_force = update_force_field_even_kernel!(backend, workgroup)
        cached_update_force_odd = update_force_field_odd_kernel!(backend, workgroup)
        cached_reset_force = reset_force_field_kernel!(backend, workgroup)
    else
        cached_update_force = nothing
        cached_update_force_odd = nothing
        cached_reset_force = nothing
    end

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
    warn_lattice_stability(ν, CType(fx), CType(fy), CType(fz), Nx, Ny, Nz; SType)

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
            SType;
            σ=CType(σ),
            σT=CType(σT),
            Tσ=Tσ === nothing ? CType(T_avg) : CType(Tσ),
            α=CType(α),
            α_s=CType(α_s),
            α_l=CType(α_l),
            ν_s=CType(ν_s),
            ν_l=CType(ν_l),
            β=CType(β),
            T_avg=CType(T_avg),
            Λ=CType(Λ),
            Ts=Ts === nothing ? CType(T_avg) : CType(Ts),
            Tl=Tl === nothing ? (Ts === nothing ? CType(T_avg) : CType(Ts)) : CType(Tl),
            K0=CType(K0),
            Λ_v=CType(Λ_v),
            T_v=CType(T_v),
            C_hk=CType(C_hk),
            p0v=CType(p0v),
            β_v=CType(β_v),
        )
    end

    buffers_ρ = [ρ(domains[d]) for d in 1:D]
    buffers_u = [u(domains[d]) for d in 1:D]
    buffers_F = [F(domains[d]) for d in 1:D]
    buffers_fi = [fi(domains[d]) for d in 1:D]
    buffers_flags = [flags(domains[d]) for d in 1:D]

    ρc = attach(buffers_ρ, Nx, Ny, Nz, Dx, Dy, Dz, "rho")
    uc = attach(buffers_u, Nx, Ny, Nz, Dx, Dy, Dz, "u")
    Fc = attach(buffers_F, Nx, Ny, Nz, Dx, Dy, Dz, "F")
    fic = attach(buffers_fi, Nx, Ny, Nz, Dx, Dy, Dz, "fi")
    fc = attach(buffers_flags, Nx, Ny, Nz, Dx, Dy, Dz, "flags")

    @static if TEMPERATURE
        buffers_T = [T(domains[d]) for d in 1:D]
        Tc = attach(buffers_T, Nx, Ny, Nz, Dx, Dy, Dz, "T")
        buffers_Q = [Q(domains[d]) for d in 1:D]
        Qc = attach(buffers_Q, Nx, Ny, Nz, Dx, Dy, Dz, "Q")
        buffers_h = [htc(domains[d]) for d in 1:D]
        hc = attach(buffers_h, Nx, Ny, Nz, Dx, Dy, Dz, "h")
        buffers_fs = [fs(domains[d]) for d in 1:D]
        fsc = attach(buffers_fs, Nx, Ny, Nz, Dx, Dy, Dz, "fs")
    end

    @static if SURFACE
        buffers_ϕ = [ϕ(domains[d]) for d in 1:D]
        ϕc = attach(buffers_ϕ, Nx, Ny, Nz, Dx, Dy, Dz, "phi")
        buffers_msrc = [msrc(domains[d]) for d in 1:D]
        msrcc = attach(buffers_msrc, Nx, Ny, Nz, Dx, Dy, Dz, "msrc")
    end

    @static if SURFACE
        @static if TEMPERATURE
            Model(
                scheme,
                backend, workgroup,
                Nx, Ny, Nz,
                Dx, Dy, Dz,
                domains,
                ρc, uc, Fc, fic, fc,
                Tc, Qc, hc, fsc,
                ϕc, msrcc,
                cached_surface_0_even,
                cached_surface_0_odd,
                cached_surface_1,
                cached_surface_2_even,
                cached_surface_2_odd,
                cached_surface_3,
                w, c,
                cached_collide_even,
                cached_collide_odd,
                cached_initialize,
                cached_moments_even,
                cached_moments_odd,
                cached_moving,
                cached_update_force,
                cached_update_force_odd,
                cached_reset_force,
                false,
                Units{CType}()
            )
        else
            Model(
                scheme,
                backend, workgroup,
                Nx, Ny, Nz,
                Dx, Dy, Dz,
                domains,
                ρc, uc, Fc, fic, fc,
                ϕc, msrcc,
                cached_surface_0_even,
                cached_surface_0_odd,
                cached_surface_1,
                cached_surface_2_even,
                cached_surface_2_odd,
                cached_surface_3,
                w, c,
                cached_collide_even,
                cached_collide_odd,
                cached_initialize,
                cached_moments_even,
                cached_moments_odd,
                cached_moving,
                cached_update_force,
                cached_update_force_odd,
                cached_reset_force,
                false,
                Units{CType}()
            )
        end
    else
        @static if TEMPERATURE
            Model(
                scheme,
                backend, workgroup,
                Nx, Ny, Nz,
                Dx, Dy, Dz,
                domains,
                ρc, uc, Fc, fic, fc,
                Tc, Qc, hc, fsc,
                w, c,
                cached_collide_even,
                cached_collide_odd,
                cached_initialize,
                cached_moments_even,
                cached_moments_odd,
                cached_moving,
                cached_update_force,
                cached_update_force_odd,
                cached_reset_force,
                false,
                Units{CType}()
            )
        else
            Model(
                scheme,
                backend, workgroup,
                Nx, Ny, Nz,
                Dx, Dy, Dz,
                domains,
                ρc, uc, Fc, fic, fc,
                w, c,
                cached_collide_even,
                cached_collide_odd,
                cached_initialize,
                cached_moments_even,
                cached_moments_odd,
                cached_moving,
                cached_update_force,
                cached_update_force_odd,
                cached_reset_force,
                false,
                Units{CType}()
            )
        end
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

@static if TEMPERATURE
    Q(model::Model) = model.Q
    htc(model::Model) = model.h
    fs(model::Model) = model.fs
    thermal_k(model::Model) = thermal_k(model.domains[1])
    thermal_k_s(model::Model) = thermal_k_s(model.domains[1])
    thermal_k_l(model::Model) = thermal_k_l(model.domains[1])
end

@static if SURFACE
    σ(model::Model) = model.domains[1].σ
    msrc(model::Model) = model.msrc
end

function warn_lattice_stability(
    ν, fx, fy, fz, Nx, Ny, Nz;
    SType::Type = Float32,
    u = nothing,
    H = nothing,
)
    C = typeof(float(ν))
    νc = C(ν)
    cs = C(1) / sqrt(C(3))
    τ = C(3) * νc + C(1) / C(2)
    ω = one(C) / τ
    fmag = hypot(C(fx), C(fy), C(fz))
    L = C(max(Int(Nx), Int(Ny), Int(Nz)))
    Hcells = H === nothing ? L : C(H)
    u_g = (fmag > 0 && Hcells > 0) ? sqrt(fmag * Hcells) : zero(C)
    u_char = u === nothing ? u_g : C(u)
    Ma = u_char / cs

    if !(νc > 0)
        @warn "lattice ν ≤ 0 is invalid" ν=νc
    end
    if !(τ > C(0.5)) || ω >= C(2)
        @warn "unstable: τ ≤ 1/2 (ω⁺ ≥ 2)" ν=νc τ ω
    elseif TRT
        ωm = one(C) / (C(0.1875) / (one(C)/ω - C(0.5)) + C(0.5))
        if ω > C(1.99)
            @info "TRT: ω⁺=$(round(Float64(ω); digits=5)) close to 2, ω⁻=$(round(Float64(ωm); digits=4)) (Λ=3/16)"
        end
    else
        if ω > C(1.99)
            @warn "lattice ν is tiny: ω=$(round(Float64(ω); digits=5)) is extremely close to 2. Increase ν or refine the grid." ν=νc τ ω
        elseif ω > C(1.95)
            @warn "lattice ω=$(round(Float64(ω); digits=4)) is close to 2; SRT is stiff" ν=νc τ ω
        end
    end
    if SType === Float16 && ω > C(1.8)
        @warn "SType=Float16 with ω=$(round(Float64(ω); digits=4)) is a common SURFACE NaN source; use Float32 until the case is stable" ω
    end
    if Ma > C(0.3)
        @warn "characteristic Mach=$(round(Float64(Ma); digits=3)) (u=$u_char, cs=$cs) is likely unstable; lower lbm_u or |g|"
    elseif Ma > C(0.15)
        @warn "characteristic Mach=$(round(Float64(Ma); digits=3)) is high for D3Q19 (target ≲ 0.1)" u=u_char
    end
    if fmag > C(1e-3)
        @warn "lattice |f|=$fmag is large; expect compressibility / SURFACE blow-up"
    end
    return nothing
end

function export!(model::Model; dir::AbstractString="output")
    start_run_log!(dir)
    model.initialized || initialize!(model)
    moments!(model)

    mkpath(dir)

    domain = model.domains[1] # not general yet
    t = Int(domain.t)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)

    ρ_host     = Array(domain.ρ.data)
    u_host     = Array(domain.u.data)
    flags_host = Array(domain.flags.data)

    umax = maximum(@views hypot.(u_host[:, 1], u_host[:, 2], u_host[:, 3]))
    if any(!isfinite, ρ_host) || any(!isfinite, u_host)
        @warn "non-finite ρ/u at t=$t - simulation has likely diverged"
    elseif umax > 0.4f0
        @warn "max |u|=$umax at t=$t exceeds ≈0.4 (cs=$(1/sqrt(3))); unstable"
    elseif umax > 0.15f0
        @warn "max |u|=$umax at t=$t is high (Ma=$(umax * sqrt(3f0)))"
    end

    U = model.units
    dx = Float32(U.m)
    t_si = si_t(U, t)

    xs = range(0f0, step=dx, length=Nx)
    ys = range(0f0, step=dx, length=Ny)
    zs = range(0f0, step=dx, length=Nz)

    ρ3     = reshape(Float32.(si_ρ.(Ref(U), ρ_host)), Nx, Ny, Nz)
    p3     = reshape(Float32.(si_p.(Ref(U), ρ_host)), Nx, Ny, Nz)
    ux     = reshape(Float32.(si_u.(Ref(U), view(u_host, :, 1))), Nx, Ny, Nz)
    uy     = reshape(Float32.(si_u.(Ref(U), view(u_host, :, 2))), Nx, Ny, Nz)
    uz     = reshape(Float32.(si_u.(Ref(U), view(u_host, :, 3))), Nx, Ny, Nz)
    flags3 = reshape(flags_host, Nx, Ny, Nz)

    pvd_path = joinpath(dir, "lbm")
    pvd = paraview_collection(pvd_path; append = isfile(pvd_path * ".pvd"))

    vtk_grid(joinpath(dir, @sprintf("lbm_%08d", t)), xs, ys, zs) do vtk
        vtk["rho"] = ρ3
        vtk["p"] = p3
        vtk["u"] = (ux, uy, uz)
        vtk["flags"] = flags3
        @static if SURFACE
            vtk["phi"] = reshape(Float32.(Array(domain.ϕ.data)), Nx, Ny, Nz)
            vtk["S"] = reshape(Float32.(si_S.(Ref(U), Array(domain.msrc.data), ρ_host)), Nx, Ny, Nz)
        end
        @static if TEMPERATURE
            vtk["T"] = reshape(Float32.(si_T.(Ref(U), Array(domain.T.data))), Nx, Ny, Nz)
            vtk["Q"] = reshape(Float32.(si_Q.(Ref(U), Array(domain.Q.data), ρ_host)), Nx, Ny, Nz)
            vtk["fs"] = reshape(Float32.(Array(domain.fs.data)), Nx, Ny, Nz)
        end
        pvd[t_si] = vtk
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
                Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz),
                domain.gi.data, domain.T.data, domain.fs.data;
                ndrange = N
            )
        else
            kernel(
                domain.ρ.data,
                domain.u.data,
                domain.fi.data,
                domain.flags.data,
                model.weights, model.velocities,
                Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz),
                domain.gi.data, domain.T.data;
                ndrange = N
            )
        end
        @static if MOVING_BOUNDARIES
            model.cached_moving!(
                domain.u.data, domain.flags.data, model.velocities,
                Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
                ndrange = N)
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
               domain.mass.data, domain.massex.data, domain.ϕ.data, domain.T.data,
               domain.fs.data,
               model.weights, model.velocities,
               domain.fx, domain.fy, domain.fz, domain.σ, domain.σT, domain.Tσ,
               domain.Λ_v, domain.T_v, domain.p0v, domain.β_v,
               Nd, Nx, Ny, Nz; ndrange = N)
        end

        @static if MOVING_BOUNDARIES
            model.cached_moving!(
                domain.u.data, domain.flags.data, model.velocities,
                Nd, Nx, Ny, Nz; ndrange = N)
        end

        kernel = t_odd ? model.cached_collide_odd! : model.cached_collide_even!
        @static if SURFACE
            kernel(domain.flags.data, domain.fi.data,
                   domain.ρ.data, domain.u.data, domain.F.data, domain.mass.data,
                   domain.gi.data, domain.T.data, domain.Q.data, domain.h.data,
                   domain.ϕ.data, domain.fs.data, domain.msrc.data,
                   model.weights, model.velocities,
                   domain.ω, domain.fx, domain.fy, domain.fz,
                   domain.ω_T, domain.β, domain.T_avg, domain.σT,
                   domain.Λ, domain.Ts, domain.Tl, domain.K0,
                   domain.α_s, domain.α_l, domain.ν_s, domain.ν_l,
                   domain.Λ_v, domain.T_v, domain.C_hk, domain.p0v, domain.β_v,
                   Nd, Nx, Ny, Nz; ndrange = N)
        else
            kernel(domain.flags.data, domain.fi.data,
                   domain.ρ.data, domain.u.data, domain.F.data,
                   domain.gi.data, domain.T.data, domain.Q.data, domain.h.data,
                   domain.fs.data,
                   model.weights, model.velocities,
                   domain.ω, domain.fx, domain.fy, domain.fz,
                   domain.ω_T, domain.β, domain.T_avg,
                   domain.Λ, domain.Ts, domain.Tl, domain.K0,
                   domain.α_s, domain.α_l, domain.ν_s, domain.ν_l,
                   domain.Λ_v, domain.T_v, domain.C_hk, domain.p0v, domain.β_v,
                   Nd, Nx, Ny, Nz; ndrange = N)
        end

        @static if SURFACE
            model.cached_surface_1!(domain.flags.data, model.velocities,
                Nd, Nx, Ny, Nz; ndrange = N)
            s2 = t_odd ? model.cached_surface_2_odd! : model.cached_surface_2_even!
            s2(domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
               domain.gi.data, domain.T.data, domain.fs.data,
               model.weights, model.velocities, Nd, Nx, Ny, Nz; ndrange = N)
            model.cached_surface_3!(
                domain.ρ.data, domain.flags.data, domain.mass.data,
                domain.massex.data, domain.ϕ.data, domain.fs.data, model.velocities,
                Nd, Nx, Ny, Nz; ndrange = N)
        end

        increment_time_step!(domain, 1)
    end
    KernelAbstractions.synchronize(model.backend)
end

@inline last_collide_odd(domain::Domain) = Int(domain.t) == 0 ? false : isodd(Int(domain.t) - 1)

function moments!(model::Model)
    for domain in model.domains
        N = get_N(domain)
        kernel = last_collide_odd(domain) ? model.cached_moments_odd! : model.cached_moments_even!
        kernel(
            domain.ρ.data,
            domain.u.data,
            domain.flags.data,
            domain.fi.data,
            domain.gi.data, domain.T.data,
            model.weights, model.velocities,
            Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
            ndrange = N
        )
    end
    KernelAbstractions.synchronize(model.backend)
end

function reset_force_field!(model::Model)
    for domain in model.domains
        fill!(domain.F.data, zero(eltype(domain.F.data)))
    end
    KernelAbstractions.synchronize(model.backend)
    return nothing
end

function update_force_field!(model::Model)
    @static if !FORCE_FIELD
        return nothing
    end
    for domain in model.domains
        N = get_N(domain)
        kernel = last_collide_odd(domain) ? model.cached_update_force_odd! : model.cached_update_force_even!
        kernel(
            domain.flags.data, domain.fi.data, domain.F.data,
            model.velocities,
            Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
            ndrange = N
        )
    end
    KernelAbstractions.synchronize(model.backend)
    return nothing
end

@static if SURFACE

@inline function surface_0_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, mass, massex, ϕ, T, fs,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
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

    frozen = false
    @static if TEMPERATURE
        frozen = is_solid_fraction(fs[n])
    end

    massn = mass[n]
    if !frozen
        for i in 2:Q
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            massn += massex[src]
        end
    end

    NP = (Q - 1) ÷ 2
    fn1 = CType(fi[f_index(n, 1, N)])

    if su == TYPE_F
        if !frozen
            for k in 1:NP
                i = 2k
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                fp_in,  fm_in  = load_pair(fi, n, src, i, t_odd, N, CType)
                fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
                massn += (fp_in - fp_out) + (fm_in - fm_out)
            end
        end
        mass[n] = massn
        return nothing
    end

    if su != TYPE_I
        mass[n] = massn
        return nothing
    end

    # TYPE_I
    cs = CType(1) / sqrt(CType(3))
    @static if EQUILIBRIUM_BOUNDARIES
        eq = (flagsn & TYPE_BO) == TYPE_E
    else
        eq = false
    end
    if frozen
        ρn = ρ[n]
        ρn = ρn > zero(CType) ? ρn : one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        uxg = zero(CType); uyg = zero(CType); uzg = zero(CType)
        ρ_gas = one(CType)
        ϕin = calculate_phi(ρn, massn, flagsn)
    elseif eq
        ρn, ux, uy, uz = prescribed_hydro(ρ[n], u[n, 1], u[n, 2], u[n, 3], fx, fy, fz)
        ϕin = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_gas = gas_density_plic(σn, ϕ, ϕin, x, y, z, Nx, Ny, Nz)
        uxg, uyg, uzg = ux, uy, uz
    else
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
        ux = clamp(ux, -cs, cs); uy = clamp(uy, -cs, cs); uz = clamp(uz, -cs, cs)
        ϕin = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_gas = gas_density_plic(σn, ϕ, ϕin, x, y, z, Nx, Ny, Nz)
        @static if VOLUME_FORCE
            uxg = clamp(ux + fx / (CType(2) * ρn), -cs, cs)
            uyg = clamp(uy + fy / (CType(2) * ρn), -cs, cs)
            uzg = clamp(uz + fz / (CType(2) * ρn), -cs, cs)
        else
            uxg, uyg, uzg = ux, uy, uz
        end
        @static if TEMPERATURE
            inv2ρ = one(CType) / (CType(2) * ρn)
            if σT != zero(CType)
                mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                uxg = clamp(uxg + mx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + my * inv2ρ, -cs, cs)
                uzg = clamp(uzg + mz * inv2ρ, -cs, cs)
            end
            if Λ_v > zero(CType)
                rx, ry, rz = recoil_force(T, ϕ, n, x, y, z, Nx, Ny, Nz, Λ_v, T_v, p0v, β_v, CType)
                uxg = clamp(uxg + rx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + ry * inv2ρ, -cs, cs)
                uzg = clamp(uzg + rz * inv2ρ, -cs, cs)
            end
        end
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

        if !frozen
            if (sup & (TYPE_F | TYPE_I)) != 0x00
                fluxp = fm_in - fp_out
                massn += sup == TYPE_F ? fluxp : CType(0.5) * (ϕp + ϕin) * fluxp
            end
            if (sum_ & (TYPE_F | TYPE_I)) != 0x00
                fluxm = fp_in - fm_out
                massn += sum_ == TYPE_F ? fluxm : CType(0.5) * (ϕm + ϕin) * fluxm
            end
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
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(false), fi, ρ, u, flags, mass, massex, ϕ, T, fs, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n))
end

@kernel function surface_0_odd_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(true), fi, ρ, u, flags, mass, massex, ϕ, T, fs, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n))
end

end