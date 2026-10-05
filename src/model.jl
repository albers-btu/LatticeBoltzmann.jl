using Printf, CUDA, WriteVTK

mutable struct Model{
    CType<:AbstractFloat,                   # Compute Type, default is Float32
    SType<:AbstractFloat,                   # Store Type, default is Float32
    Aρ<:AbstractArray{CType},               # Array for ρ (density)
    Au<:AbstractArray{CType},               # Array for u (velocity)
    Afi<:AbstractArray{SType},              # Array for fᵢ DDF (discrete distribution function)
    Af<:AbstractArray{UInt8},               # Array for flags
    Q                                       # Number of discrete velocities: DdQ19 leads to Q = 19
}
    scheme::Symbol                          # LBM Scheme, default is D3Q19

    backend::KernelAbstractions.Backend     # Backend for the kernel execution, CPU() (default) or CUDABackend()
    workgroup::Int                          # Determines block size on GPU (threads per block) and loop chunk size on CPU

    Nx::UInt                                # Lattice size x
    Ny::UInt                                # Lattice size y
    Nz::UInt                                # Lattice size z

    Dx::UInt                                # Domain size x
    Dy::UInt                                # Domain size y
    Dz::UInt                                # Domain size z

    domains::Vector{<:Domain{CType, SType}} # Store of domains for this model

    ρ::MemoryContainer{CType, Aρ}           # View of the ρ (density) memory
    u::MemoryContainer{CType, Au}           # View of the u (velocity) memory
    F::MemoryContainer{CType, Au}           # View of the f (force) memory
    fi::MemoryContainer{SType, Afi}         # View of the fᵢ (DDF) memory
    flags::MemoryContainer{UInt8, Af}       # View of the flags memory

    @static if TEMPERATURE
        T::MemoryContainer{CType, Aρ}       # View of T (lattice temperature) memory
        Q::MemoryContainer{CType, Aρ}       # View of Q (volumetric heat) memory
        h::MemoryContainer{CType, Aρ}       # View of h (Robin BC coefficient, 0 leads to Neumann BC) memory
        fs::MemoryContainer{CType, Aρ}      # View of fs (solid fraction) memory
    end

    @static if SURFACE
        ϕ::MemoryContainer{CType, Aρ}       # View of ϕ (liquid fill fraction) memory
        msrc::MemoryContainer{CType, Aρ}    # View of msrc (mass source) memory
        cached_surface_0_even_kernel!::Any  # Cached kernel
        cached_surface_0_odd_kernel!::Any   # Cached kernel
        cached_surface_1_kernel!::Any       # Cached kernel
        cached_surface_2_even_kernel!::Any  # Cached kernel
        cached_surface_2_odd_kernel!::Any   # Cached kernel
        cached_surface_3_kernel!::Any       # Cached kernel
    end

    weights::NTuple{Q, CType}               # Scheme weights
    velocities::NTuple{Q, SVector{3, Int}}  # Scheme velocities

    cached_collide_even_kernel!::Any        # Cached kernel
    cached_collide_odd_kernel!::Any         # Cached kernel
    cached_initialize_kernel!::Any          # Cached kernel
    cached_moments_even_kernel!::Any        # Cached kernel
    cached_moments_odd_kernel!::Any         # Cached kernel
    cached_moving_kernel!::Any              # Cached kernel
    cached_update_force_even_kernel!::Any   # Cached kernel
    cached_update_force_odd_kernel!::Any    # Cached kernel
    cached_reset_force_kernel!::Any         # Cached kernel

    initialized::Bool                       # Flag to indicate finished initialization
    units::Units{CType}                     # Units to convert between lattice and SI
    laser::Any                              # Laser source (see laser.jl)
    powder_jet::Any                         # Powder source (see powder.jl)
    n_hydro::Int                            # Hydro/interface substeps per thermal step
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
    k_sT = 0.0,
    k_lT = 0.0,
    cp_sT = 0.0,
    cp_lT = 0.0,
    ν_sT = 0.0,
    ν_lT = 0.0,
    β = 0.0f0,
    T_avg = 1.0f0,
    latent = 0.0f0,
    Ts = nothing,
    Tl = nothing,
    K0 = 0.0f0,                             # Kozeny-Carman permeability
    latent_v = 0.0f0,
    T_v = nothing,
    M = 0.0558,                             # Molar mass (kg/mol) of 316L, 55.8 kg/mol
    p_atm = 101325.0,
    emissivity = 0.0,
    T_rad = nothing,                        # Far-field temperature for radiation
    powder_τ = 0.0,                         # Unmelted powder lifetime 
    powder_T = nothing,                     # Powder temperature
    laser = nothing,
    powder_jet = nothing,
    n_hydro::Int = 1,
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
    α  = CType(lbm_α(units, α))
    αs = α_s === nothing ? α : CType(lbm_α(units, α_s))
    αl = α_l === nothing ? α : CType(lbm_α(units, α_l))
    νs = ν_s === nothing ? ν : CType(lbm_ν(units, ν_s))
    νl = ν_l === nothing ? ν : CType(lbm_ν(units, ν_l))
    αsT = CType(lbm_αT(units, k_sT))
    αlT = CType(lbm_αT(units, k_lT))
    γs = CType(lbm_γ(units, cp_sT))
    γl = CType(lbm_γ(units, cp_lT))
    νsT = CType(lbm_νT(units, ν_sT))
    νlT = CType(lbm_νT(units, ν_lT))
    Λ  = CType(lbm_Λ(units, latent))
    Tsl = Ts === nothing ? CType(T_avg) :
          Ts isa Quantity ? CType(lbm_T(units, Ts)) : CType(Ts)
    Tll = Tl === nothing ? CType(T_avg) :
          Tl isa Quantity ? CType(lbm_T(units, Tl)) : CType(Tl)
    Tvl = T_v === nothing ? zero(CType) :
          T_v isa Quantity ? CType(lbm_T(units, T_v)) : CType(T_v)
    K0l = K0 isa Quantity ? CType(ustrip(u"m^2", K0) / units.m^2) : CType(K0 / units.m^2)
    Λv, βv, p0l, Chk = lbm_evap(units, latent_v, M, p_atm)
    Lvsi = latent_v isa Quantity ? ustrip(u"J/kg", latent_v) : Float64(latent_v)
    if Lvsi > 0 && Tvl == 0
        @warn "latent_v > 0 but T_v is unset; evaporation will stay off"
        Λv = zero(CType)
    end
    τp = powder_τ isa Quantity ? CType(ustrip(u"s", powder_τ) / units.s) : CType(powder_τ)
    Tp = powder_T === nothing ? CType(T_avg) :
         powder_T isa Quantity ? CType(lbm_T(units, powder_T)) : CType(powder_T)
    εr = emissivity isa Quantity ? ustrip(emissivity) : Float64(emissivity)
    Crad = εr > 0 ? CType(lbm_rad(units, εr)) : zero(CType)
    Trad = T_rad === nothing ? CType(T_avg) :
           T_rad isa Quantity ? CType(lbm_T(units, T_rad)) : CType(T_rad)

    model = Model(Nx, Ny, Nz, ν; fx, fy, fz, σ=σ, σT=σT, Tσ=Tσl, α=α, α_s=αs, α_l=αl,
                  ν_s=νs, ν_l=νl, α_sT=αsT, α_lT=αlT, γ_s=γs, γ_l=γl, ν_sT=νsT, ν_lT=νlT,
                  β=CType(β), T_avg=CType(T_avg),
                  Λ=Λ, Ts=Tsl, Tl=Tll, K0=K0l,
                  Λ_v=CType(Λv), T_v=Tvl, C_hk=CType(Chk), p0v=CType(p0l), β_v=CType(βv),
                  C_rad=Crad, T_rad=Trad,
                  τ_p=τp, T_p=Tp,
                  laser=laser, powder_jet=powder_jet, n_hydro=n_hydro, CType, SType, scheme, backend, workgroup)
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
    α_sT = 0.0f0,
    α_lT = 0.0f0,
    γ_s = 0.0f0,
    γ_l = 0.0f0,
    ν_s = 0.0f0,
    ν_l = 0.0f0,
    ν_sT = 0.0f0,
    ν_lT = 0.0f0,
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
    C_rad = 0.0f0,
    T_rad = nothing,
    τ_p = 0.0f0,
    T_p = nothing,
    laser = nothing,
    powder_jet = nothing,
    n_hydro::Int = 1,
    CType::Type{<:AbstractFloat} = Float32,
    SType::Type{<:AbstractFloat} = CType,
    scheme = :D3Q19, 
    backend = CPU(), 
    workgroup = default_workgroup(backend)
)
    backend isa CUDABackend && !CUDA.functional() && throw(ArgumentError("CUDABackend requested but CUDA is not functional"))

    @static if DIM == 2
        Int(Nz) == 1 || throw(ArgumentError("DIM=2 requires Nz == 1, got $(Int(Nz))"))
        if scheme == :D3Q19
            scheme = :D2Q9
        elseif scheme != :D2Q9
            throw(ArgumentError("DIM=2 scheme must be :D2Q9 or :D3Q19, got $scheme"))
        end
        if fz != 0
            @warn "DIM=2 ignores gz; use gy for in-plane gravity"
            fz = 0
        end
    end

    w = weights(scheme, CType)
    c = velocities(scheme)
    Q_T = @static DIM == 3 ? 7 : 5
    @info "lattice image" DIM SCHEME SCHEME_T scheme Q=length(w) Q_T

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
    @static if DIM == 2
        Int(Dz) == 1 || throw(ArgumentError("DIM=2 requires Dz == 1, got $(Int(Dz))"))
    end
    D = UInt(Dx*Dy*Dz)

    if Nx % Dx != 0 || Ny % Dy != 0 || Nz % Dz != 0
        @warn "grid is not equally divisible in domains"
    end

    Nx::UInt = UInt(Nx)
    Ny::UInt = UInt(Ny)
    Nz::UInt = UInt(Nz)

    Hx::UInt = UInt(Dx > 1) # Halo offset x
    Hy::UInt = UInt(Dy > 1) # Halo offset y
    Hz::UInt = UInt(Dz > 1) # Halo offset z
    
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
            α_sT=CType(α_sT),
            α_lT=CType(α_lT),
            γ_s=CType(γ_s),
            γ_l=CType(γ_l),
            ν_s=CType(ν_s),
            ν_l=CType(ν_l),
            ν_sT=CType(ν_sT),
            ν_lT=CType(ν_lT),
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
            C_rad=CType(C_rad),
            T_rad=T_rad === nothing ? CType(T_avg) : CType(T_rad),
            τ_p=CType(τ_p),
            T_p=T_p === nothing ? CType(T_avg) : CType(T_p),
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
                Units{CType}(),
                laser,
                powder_jet,
                max(1, n_hydro),
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
                Units{CType}(),
                laser,
                powder_jet,
                max(1, n_hydro),
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
                Units{CType}(),
                laser,
                powder_jet,
                max(1, n_hydro),
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
                Units{CType}(),
                laser,
                powder_jet,
                max(1, n_hydro),
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
    enthalpy(model::Model) = mapreduce(enthalpy, +, model.domains)
    heat_source(model::Model) = mapreduce(heat_source, +, model.domains)
    reset_energy_budget!(model::Model) = (foreach(reset_energy_budget!, model.domains); model)
    function energy_budget(model::Model)
        parts = map(energy_budget, model.domains)
        H = sum(p -> p.H, parts)
        H0 = sum(p -> p.H0, parts)
        Q = sum(p -> p.Q, parts)
        rad = sum(p -> p.rad, parts)
        evap = sum(p -> p.evap, parts)
        wall = sum(p -> p.wall, parts)
        powder = sum(p -> p.powder, parts)
        expected = H0 + Q - rad - evap + powder - wall
        return (; H, H0, Q, rad, evap, wall, powder, expected, residual=H - expected)
    end
    @static if SURFACE
        metal_mass(model::Model) = mapreduce(metal_mass, +, model.domains)
        reset_mass_budget!(model::Model) = (foreach(reset_mass_budget!, model.domains); model)
        function mass_budget(model::Model)
            parts = map(mass_budget, model.domains)
            M = sum(p -> p.M, parts)
            M0 = sum(p -> p.M0, parts)
            evap = sum(p -> p.evap, parts)
            powder = sum(p -> p.powder, parts)
            expected = M0 + powder - evap
            return (; M, M0, evap, powder, expected, residual=M - expected)
        end
    end
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
    L = C(max(Int(Nx), Int(Ny), Int(Nz)))                           # Largest grid size in cells
    Hcells = H === nothing ? L : C(H)                               # Height in cells
    u_g = (fmag > 0 && Hcells > 0) ? sqrt(fmag * Hcells) : zero(C)  # Characteristic speed of falling under gravity
    u_char = u === nothing ? u_g : C(u)
    Ma = u_char / cs

                                                                    # Positive kinematic viscosity
    if !(νc > 0)                                                    # ν = 1/3 * (τ - 1/2)
        @warn "lattice ν ≤ 0 is invalid" ν=νc
    end
    if τ <= C(0.5) || ω >= C(2)                                     # τ leads to unstable behaviour, negative viscosity
        @warn "unstable: τ ≤ 1/2 (ω⁺ ≥ 2)" ν=νc τ ω
    elseif TRT
        ωm = one(C) / (C(0.1875) / (one(C)/ω - C(0.5)) + C(0.5))    # ω⁻ = 1 / (Λ / (1/ω⁺ - 0.5)) + 0.5)
                                                                    # using Λ = 3/16 ≈ 0.1875
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
        @static if DIM == 3
            @warn "characteristic Mach=$(round(Float64(Ma); digits=3)) is high for D3Q19 (target ≲ 0.1)" u=u_char
        else
            @warn "characteristic Mach=$(round(Float64(Ma); digits=3)) is high for D2Q9 (target ≲ 0.1)" u=u_char
        end
    end
    if fmag > C(1e-3)
        @warn "lattice |f|=$fmag is large; expect compressibility / SURFACE blow-up"
    end
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
    kernel = model.cached_initialize_kernel!
    for domain in model.domains
        N = get_N(domain)
        @static if SURFACE
            kernel(
                domain.ρ.data,
                domain.u.data,
                domain.fi.data,
                domain.flags.data,
                domain.mass.data, domain.massex.data, domain.ϕ.data,                # SURFACE specific
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
            model.cached_moving_kernel!(
                domain.u.data, domain.flags.data, model.velocities,
                Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz);
                ndrange = N
            )
        end
    end

    KernelAbstractions.synchronize(model.backend)
    model.initialized = true
    @static if TEMPERATURE
        reset_energy_budget!(model)
        @static if SURFACE
            reset_mass_budget!(model)
        end
    end
    @info "finished init"
end

function step!(model::Model)
    nsub = max(1, model.n_hydro)
    invN = 1 / nsub
    invN2 = invN * invN
    for domain in model.domains
        N = get_N(domain)
        Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
        Nd = Int(domain.N)
        # Coefficients on Domain are for one outer step (Units.s).
        # Hydro substep Δt is that / n_hydro: ν ∝ Δt, σ, g, p ∝ Δt².
        CT = eltype(domain.ν)
        sν = CT(invN)
        s2 = CT(invN2)
        fx = domain.fx * s2; fy = domain.fy * s2; fz = domain.fz * s2
        σ = domain.σ * s2
        σT = domain.σT * s2
        p0v = domain.p0v * s2
        νs = domain.ν_s * sν; νl = domain.ν_l * sν
        νsT = domain.ν_sT * sν; νlT = domain.ν_lT * sν
        ω = omega_from_nu(domain.ν * sν)

        @static if SURFACE && TEMPERATURE
            deposit_laser!(model, domain)
            advance_powder_jet!(model, domain)
            if domain.τ_p > 0
                powder_gas_kernel!(model.backend, model.workgroup)(
                                   domain.flags.data, domain.mp.data, domain.msrc.data, domain.ρ.data,
                                   domain.τ_p, domain.T_p, domain.γ_s,
                                   domain.Eacc.data, domain.Macc.data, Nd; ndrange = N)
            end
        end

        # gi streams once per outer step. Init writes it for an even load,
        # so that load is isodd(t), not the hydro substep index.
        g_odd = isodd(Int(domain.t))
        for sub in 1:nsub
            # fi parity advances every hydro substep and stays continuous across outer steps.
            t_odd = isodd(Int(domain.t) * nsub + sub - 1)
            thermal = sub == nsub
            # store(P) is pulled by load(!P). A cell born before the thermal
            # collide must be visible to load(g_odd); one born after it, to load(!g_odd).
            g_store_odd = thermal ? g_odd : !g_odd

            @static if SURFACE
                sk0 = t_odd ? model.cached_surface_0_odd_kernel! : model.cached_surface_0_even_kernel!
                sk0(domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
                    domain.mass.data, domain.massex.data, domain.ϕ.data, domain.T.data,
                    domain.fs.data, domain.gi.data,
                    model.weights, model.velocities,
                    fx, fy, fz, σ, σT, domain.Tσ,
                    domain.Λ_v, domain.T_v, p0v, domain.β_v,
                    Nd, Nx, Ny, Nz, domain.Eacc.data,
                    domain.h.data, domain.Q.data, domain.ω_T,
                    thermal, g_odd; ndrange = N)
            end

            @static if MOVING_BOUNDARIES
                model.cached_moving_kernel!(
                    domain.u.data, domain.flags.data, model.velocities,
                    Nd, Nx, Ny, Nz; ndrange = N)
            end

            kernel = t_odd ? model.cached_collide_odd_kernel! : model.cached_collide_even_kernel!
            @static if SURFACE
                kernel(domain.flags.data, domain.fi.data,
                       domain.ρ.data, domain.u.data, domain.F.data,
                       domain.mass.data,
                       domain.gi.data, domain.T.data, domain.Q.data, domain.h.data,
                       domain.ϕ.data,
                       domain.fs.data,
                       domain.msrc.data, domain.mp.data,
                       model.weights, model.velocities,
                       ω, fx, fy, fz,
                       domain.ω_T, domain.β, domain.T_avg, σT,
                       domain.Λ, domain.Ts, domain.Tl, domain.K0,
                       domain.α_s, domain.α_l, domain.α_sT, domain.α_lT,
                       domain.γ_s, domain.γ_l,
                       νs, νl, νsT, νlT,
                       domain.Λ_v, domain.T_v, domain.C_hk, p0v, domain.β_v,
                       domain.C_rad, domain.T_rad, domain.τ_p, domain.T_p,
                       thermal, g_odd,
                       Nd, Nx, Ny, Nz, domain.Eacc.data,
                       domain.Macc.data;
                       ndrange = N)
            else
                kernel(domain.flags.data, domain.fi.data,
                       domain.ρ.data, domain.u.data, domain.F.data,
                       domain.gi.data, domain.T.data, domain.Q.data, domain.h.data,
                       domain.fs.data,
                       model.weights, model.velocities,
                       ω, fx, fy, fz,
                       domain.ω_T, domain.β, domain.T_avg,
                       domain.Λ, domain.Ts, domain.Tl, domain.K0,
                       domain.α_s, domain.α_l, domain.α_sT, domain.α_lT,
                       domain.γ_s, domain.γ_l,
                       νs, νl, νsT, νlT,
                       domain.Λ_v, domain.T_v, domain.C_hk, p0v, domain.β_v,
                       domain.C_rad, domain.T_rad,
                       thermal, g_odd,
                       Nd, Nx, Ny, Nz, domain.Eacc.data; ndrange = N)
            end

            @static if SURFACE
                model.cached_surface_1_kernel!(domain.flags.data, model.velocities,
                                               Nd, Nx, Ny, Nz; ndrange = N)
                sk2 = t_odd ? model.cached_surface_2_odd_kernel! : model.cached_surface_2_even_kernel!
                sk2(domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
                    domain.gi.data, domain.T.data, domain.fs.data,
                    model.weights, model.velocities, Nd, Nx, Ny, Nz, g_store_odd; ndrange = N)
                model.cached_surface_3_kernel!(domain.ρ.data, domain.flags.data, domain.mass.data,
                                               domain.massex.data, domain.ϕ.data, domain.fs.data, model.velocities,
                                               Nd, Nx, Ny, Nz; ndrange = N)
            end
        end

        increment_time_step!(domain, 1)
    end
    KernelAbstractions.synchronize(model.backend)
end

@inline function last_collide_odd(model::Model, domain::Domain)
    Int(domain.t) == 0 && return false
    nsub = max(1, model.n_hydro)
    return isodd(Int(domain.t) * nsub - 1)
end

# Parity of the thermal collide that just finished. t == 0 means init, which is even.
@inline function last_thermal_odd(domain::Domain)
    t = Int(domain.t)
    t == 0 && return false
    return isodd(t - 1)
end

function moments!(model::Model)
    for domain in model.domains
        N = get_N(domain)
        kernel = last_collide_odd(model, domain) ? model.cached_moments_odd_kernel! : model.cached_moments_even_kernel!
        kernel(
            domain.ρ.data,
            domain.u.data,
            domain.flags.data,
            domain.fi.data,
            domain.gi.data, domain.T.data,
            model.weights, model.velocities,
            Int(domain.N), Int(domain.Nx), Int(domain.Ny), Int(domain.Nz),
            last_thermal_odd(domain);
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
        kernel = last_collide_odd(model, domain) ? model.cached_update_force_odd_kernel! : model.cached_update_force_even_kernel!
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