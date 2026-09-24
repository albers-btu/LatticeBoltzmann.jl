@inline function _vtk_finite32(x, default=0.0f0)
    y = Float32(x)
    return isfinite(y) ? y : default
end

@inline function _vtk_clamp32(x, lo, hi)
    y = _vtk_finite32(x)
    return y < lo ? lo : (y > hi ? hi : y)
end

function _vtk_scalar(A, Nx, Ny, Nz, f=identity; lo=nothing, hi=nothing)
    B = Array{Float32}(undef, Nx, Ny, Nz)
    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        val = _vtk_finite32(f(A[n]))
        lo !== nothing && val < lo && (val = lo)
        hi !== nothing && val > hi && (val = hi)
        B[x, y, z] = val
    end
    return B
end

function _vtk_T(T_lattice, flags, U, Nx, Ny, Nz)
    B = Array{Float32}(undef, Nx, Ny, Nz)
    Tlo = 0.0f0
    Thi = 20000.0f0
    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if (flags[n] & TYPE_SU) == TYPE_G
            B[x, y, z] = 0.0f0
        else
            B[x, y, z] = _vtk_clamp32(si_T(U, T_lattice[n]), Tlo, Thi)
        end
    end
    return B
end

function _vtk_T(T_lattice, flags, U, Nx, Ny, Nz)
    B = Array{Float32}(undef, Nx, Ny, Nz)
    Tlo = 0.0f0
    Thi = 20000.0f0
    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if (flags[n] & TYPE_SU) == TYPE_G
            B[x, y, z] = 0.0f0
        else
            B[x, y, z] = _vtk_clamp32(si_T(U, T_lattice[n]), Tlo, Thi)
        end
    end
    return B
end

function export!(model::Model; dir::AbstractString="output")
    start_run_log!(dir)
    model.initialized || initialize!(model)
    moments!(model)

    mkpath(dir)

    domain = model.domains[1]
    t = Int(domain.t)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)

    ρ_host     = Array(domain.ρ.data)
    u_host     = Array(domain.u.data)
    flags_host = Array(domain.flags.data)

    umax = maximum(@views hypot.(u_host[:, 1], u_host[:, 2], u_host[:, 3]))
    nbad_ρ = count(!isfinite, ρ_host)
    nbad_u = count(!isfinite, u_host)

    if umax > 0.4f0
        @warn "max |u|=$umax at t=$t exceeds 0.4 (cₛ = $(1/sqrt(3))); unstable"
    elseif umax > 0.15f0
        @warn "max |u|=$umax at t=$t is high (Ma = $(umax * sqrt(3f0)), try to keep below Ma ≲ 0.1)"
    end

    if nbad_ρ > 0
        @warn "non-finite ρ at t = $t ($nbad_ρ values) — writing 0 in VTK" nbad_ρ
    end
    if nbad_u > 0
        @warn "non-finite u at t = $t ($nbad_u values) — writing 0 in VTK" nbad_u
    end

    U = model.units
    dx = Float32(U.m)
    isfinite(dx) && dx > 0 || (dx = 1.0f0)
    t_si = Float64(si_t(U, t))
    isfinite(t_si) || (t_si = Float64(t))

    xs = range(0.0f0, step=dx, length=Nx)
    ys = range(0.0f0, step=dx, length=Ny)
    zs = range(0.0f0, step=dx, length=Nz)

    ρ3 = _vtk_scalar(ρ_host, Nx, Ny, Nz, ρ -> si_ρ(U, ρ); lo=0.0f0, hi=1.0f7)
    p3 = _vtk_scalar(ρ_host, Nx, Ny, Nz, ρ -> si_p(U, ρ); lo=-1.0f12, hi=1.0f12)
    ux = _vtk_scalar(view(u_host, :, 1), Nx, Ny, Nz, u -> si_u(U, u); lo=-1.0f5, hi=1.0f5)
    uy = _vtk_scalar(view(u_host, :, 2), Nx, Ny, Nz, u -> si_u(U, u); lo=-1.0f5, hi=1.0f5)
    uz = _vtk_scalar(view(u_host, :, 3), Nx, Ny, Nz, u -> si_u(U, u); lo=-1.0f5, hi=1.0f5)
    flags3 = reshape(Int32.(flags_host), Nx, Ny, Nz)

    pvd_path = joinpath(dir, "lbm")
    # t=0 starts a new collection so a re-run does not append duplicate
    # timesteps into an old .pvd
    pvd = paraview_collection(pvd_path; append = t > 0 && isfile(pvd_path * ".pvd"))

    vtk_grid(joinpath(dir, @sprintf("lbm_%08d", t)), xs, ys, zs) do vtk
        @static if TEMPERATURE
            vtk["T"] = _vtk_T(Array(domain.T.data), flags_host, U, Nx, Ny, Nz)
            vtk["fs"] = _vtk_scalar(Array(domain.fs.data), Nx, Ny, Nz; lo=0.0f0, hi=1.0f0)
        end
        @static if SURFACE
            vtk["phi"] = _vtk_scalar(Array(domain.ϕ.data), Nx, Ny, Nz; lo=0.0f0, hi=2.0f0)
        end
        vtk["u"] = (ux, uy, uz)
        vtk["rho"] = ρ3
        vtk["p"] = p3
        vtk["flags"] = flags3
        @static if SURFACE
            vtk["mp"] = _vtk_fillfrac(Array(domain.mp.data), ρ_host, Nx, Ny, Nz)
            vtk["S"] = _vtk_scalar(Array(domain.msrc.data), Nx, Ny, Nz, S -> si_S(U, S, 1); lo=-1.0f6, hi=1.0f6)
        end
        @static if TEMPERATURE
            vtk["Q"] = _vtk_scalar(Array(domain.Q.data), Nx, Ny, Nz, Qlat -> si_Q(U, Qlat, 1); lo=-1.0f15, hi=1.0f15)
            vtk[VTKPointData()] = ("Scalars" => "T", "Vectors" => "u")
        elseif SURFACE
            vtk[VTKPointData()] = ("Scalars" => "phi", "Vectors" => "u")
        else
            vtk[VTKPointData()] = ("Vectors" => "u",)
        end
        pvd[t_si] = vtk
    end

    vtk_save(pvd)
    return nothing
end