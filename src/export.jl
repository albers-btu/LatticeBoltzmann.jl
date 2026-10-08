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

function _vtk_fillfrac(mp, ρ, Nx, Ny, Nz)
    B = Array{Float32}(undef, Nx, Ny, Nz)
    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        ρn = _vtk_finite32(ρ[n])
        B[x, y, z] = ρn > 0 ? _vtk_clamp32(mp[n] / ρn, 0.0f0, 2.0f0) : 0.0f0
    end
    return B
end

# SoA slots in the packed export buffer. -1 = not written.
const VTK_RHO = 1 << 0
const VTK_P   = 1 << 1
const VTK_U   = 1 << 2
const VTK_T   = 1 << 3
const VTK_FS  = 1 << 4
const VTK_PHI = 1 << 5
const VTK_MP  = 1 << 6
const VTK_S   = 1 << 7
const VTK_Q   = 1 << 8
const VTK_FLAGS = 1 << 9

@inline function _vtk_num(x::Float32)
    return ifelse(isfinite(x), x, 0.0f0)
end

@inline function _vtk_lim(x::Float32, lo::Float32, hi::Float32)
    y = _vtk_num(x)
    return ifelse(y < lo, lo, ifelse(y > hi, hi, y))
end

# One cell per thread. buf is component-major, index n is x-fastest (VTK order).
@kernel function pack_vtk_kernel!(buf, ρ, u, flags, T, fs, ϕ, mp, msrc, Q,
        N::Int,
        off_rho::Int, off_p::Int, off_u::Int, off_T::Int, off_fs::Int,
        off_phi::Int, off_mp::Int, off_S::Int, off_Q::Int, off_flags::Int,
        ρs::Float32, ps::Float32, us::Float32, Ts::Float32, Ss::Float32, Qs::Float32)
    n = @index(Global)
    ρn = _vtk_num(Float32(ρ[n]))
    if off_rho >= 0
        buf[off_rho * N + n] = _vtk_lim(ρn * ρs, 0.0f0, 1.0f7)
    end
    if off_p >= 0
        buf[off_p * N + n] = _vtk_lim(ρn * ps, -1.0f12, 1.0f12)
    end
    if off_u >= 0
        buf[off_u * N + n]       = _vtk_lim(Float32(u[n, 1]) * us, -1.0f5, 1.0f5)
        buf[(off_u + 1) * N + n] = _vtk_lim(Float32(u[n, 2]) * us, -1.0f5, 1.0f5)
        buf[(off_u + 2) * N + n] = _vtk_lim(Float32(u[n, 3]) * us, -1.0f5, 1.0f5)
    end
    @static if TEMPERATURE
        if off_T >= 0
            fl = flags[n]
            raw = ((fl & TYPE_SU) == TYPE_G) ? 0.0f0 : Float32(T[n]) * Ts
            buf[off_T * N + n] = _vtk_lim(raw, 0.0f0, 20000.0f0)
        end
        if off_fs >= 0
            buf[off_fs * N + n] = _vtk_lim(Float32(fs[n]), 0.0f0, 1.0f0)
        end
        if off_Q >= 0
            buf[off_Q * N + n] = _vtk_lim(Float32(Q[n]) * Qs, -1.0f15, 1.0f15)
        end
    end
    @static if SURFACE
        if off_phi >= 0
            buf[off_phi * N + n] = _vtk_lim(Float32(ϕ[n]), 0.0f0, 2.0f0)
        end
        if off_mp >= 0
            buf[off_mp * N + n] = ρn > 0 ? _vtk_lim(Float32(mp[n]) / ρn, 0.0f0, 2.0f0) : 0.0f0
        end
        if off_S >= 0
            buf[off_S * N + n] = _vtk_lim(Float32(msrc[n]) * Ss, -1.0f6, 1.0f6)
        end
    end
    if off_flags >= 0
        buf[off_flags * N + n] = Float32(flags[n])
    end
end

function _vtk_mask(fields)
    mask = 0
    wanted = fields === nothing ? (:rho, :p, :u, :flags, :T, :fs, :phi, :mp, :S, :Q) : fields
    for name in wanted
        name === :rho && (mask |= VTK_RHO)
        name === :p   && (mask |= VTK_P)
        name === :u   && (mask |= VTK_U)
        name === :T   && (mask |= VTK_T)
        name === :fs  && (mask |= VTK_FS)
        name === :phi && (mask |= VTK_PHI)
        name === :mp  && (mask |= VTK_MP)
        name === :S   && (mask |= VTK_S)
        name === :Q   && (mask |= VTK_Q)
        name === :flags && (mask |= VTK_FLAGS)
    end
    @static if !TEMPERATURE
        mask &= ~(VTK_T | VTK_FS | VTK_Q)
    end
    @static if !SURFACE
        mask &= ~(VTK_PHI | VTK_MP | VTK_S)
    end
    return mask
end

function _vtk_offsets(mask::Int)
    i = 0
    function take(bit, k=1)
        if mask & bit != 0
            o = i
            i += k
            return o
        end
        return -1
    end
    offs = (
        rho = take(VTK_RHO),
        p   = take(VTK_P),
        u   = take(VTK_U, 3),
        T   = take(VTK_T),
        fs  = take(VTK_FS),
        phi = take(VTK_PHI),
        mp  = take(VTK_MP),
        S   = take(VTK_S),
        Q   = take(VTK_Q),
        flags = take(VTK_FLAGS),
    )
    return offs, i
end

# Two host slots so a write of frame k can overlap the copy of frame k+1.
mutable struct _VtkSlot
    host::Vector{Float32}
end

const _VTK_SLOTS = _VtkSlot[_VtkSlot(Float32[]), _VtkSlot(Float32[])]
const _VTK_FREE = Channel{Int}(2)
const _VTK_JOBS = Channel{Any}(2)
const _VTK_STARTED = Ref(false)
const _VTK_DEV = Ref{Any}(nothing)
const _VTK_COPY_EV = Ref{Any}(nothing)
const _VTK_STREAM = Ref{Any}(nothing)
const _VTK_TASK = Ref{Task}()

struct _VtkJob
    dir::String
    t::Int
    t_si::Float64
    dx::Float32
    Nx::Int
    Ny::Int
    Nz::Int
    host::Vector{Float32}
    offs::NamedTuple
    slot::Int
    event::Any
    ctx::Any
end

function _vtk_writer_loop()
    while true
        job = take!(_VTK_JOBS)
        job === nothing && break
        try
            if job.event !== nothing
                CUDA.context!(job.ctx) do
                    CUDA.synchronize(job.event)
                end
            end
            _vtk_write(job)
        catch err
            @error "VTK export failed" exception=(err, catch_backtrace())
        finally
            put!(_VTK_FREE, job.slot)
        end
    end
    return nothing
end

function _ensure_vtk_writer()
    _VTK_STARTED[] && return
    put!(_VTK_FREE, 1)
    put!(_VTK_FREE, 2)
    _VTK_TASK[] = Threads.@spawn _vtk_writer_loop()
    atexit(flush_exports!)
    _VTK_STARTED[] = true
    return nothing
end

function _vtk_comp(host, off, N, Nx, Ny, Nz)
    return reshape(@view(host[(off * N + 1):((off + 1) * N)]), Nx, Ny, Nz)
end

function _vtk_write(job::_VtkJob)
    host, N = job.host, job.Nx * job.Ny * job.Nz
    offs = job.offs
    if offs.u >= 0
        ux = _vtk_comp(host, offs.u, N, job.Nx, job.Ny, job.Nz)
        uy = _vtk_comp(host, offs.u + 1, N, job.Nx, job.Ny, job.Nz)
        uz = _vtk_comp(host, offs.u + 2, N, job.Nx, job.Ny, job.Nz)
        umax = maximum(hypot.(ux, uy, uz))
        if umax > 0.4f0
            @warn "max |u|=$umax at t=$(job.t) exceeds 0.4 (cₛ = $(1/sqrt(3))); unstable"
        elseif umax > 0.15f0
            @warn "max |u|=$umax at t=$(job.t) is high (Ma = $(umax * sqrt(3f0)))"
        end
    end
    xs = range(0.0f0, step=job.dx, length=job.Nx)
    ys = range(0.0f0, step=job.dx, length=job.Ny)
    zs = range(0.0f0, step=job.dx, length=job.Nz)
    pvd_path = joinpath(job.dir, "lbm")
    pvd = paraview_collection(pvd_path; append = job.t > 0 && isfile(pvd_path * ".pvd"))
    vtk_grid(joinpath(job.dir, @sprintf("lbm_%08d", job.t)), xs, ys, zs) do vtk
        offs.u >= 0 && (vtk["u"] = (
            _vtk_comp(host, offs.u, N, job.Nx, job.Ny, job.Nz),
            _vtk_comp(host, offs.u + 1, N, job.Nx, job.Ny, job.Nz),
            _vtk_comp(host, offs.u + 2, N, job.Nx, job.Ny, job.Nz)))
        offs.rho >= 0 && (vtk["rho"] = _vtk_comp(host, offs.rho, N, job.Nx, job.Ny, job.Nz))
        offs.p >= 0 && (vtk["p"] = _vtk_comp(host, offs.p, N, job.Nx, job.Ny, job.Nz))
        @static if TEMPERATURE
            offs.T >= 0 && (vtk["T"] = _vtk_comp(host, offs.T, N, job.Nx, job.Ny, job.Nz))
            offs.fs >= 0 && (vtk["fs"] = _vtk_comp(host, offs.fs, N, job.Nx, job.Ny, job.Nz))
            offs.Q >= 0 && (vtk["Q"] = _vtk_comp(host, offs.Q, N, job.Nx, job.Ny, job.Nz))
        end
        @static if SURFACE
            offs.phi >= 0 && (vtk["phi"] = _vtk_comp(host, offs.phi, N, job.Nx, job.Ny, job.Nz))
            offs.mp >= 0 && (vtk["mp"] = _vtk_comp(host, offs.mp, N, job.Nx, job.Ny, job.Nz))
            offs.S >= 0 && (vtk["S"] = _vtk_comp(host, offs.S, N, job.Nx, job.Ny, job.Nz))
        end
        if offs.flags >= 0
            vtk["flags"] = Int32.(_vtk_comp(host, offs.flags, N, job.Nx, job.Ny, job.Nz))
        end
        @static if TEMPERATURE
            vtk[VTKPointData()] = ("Scalars" => "T", "Vectors" => "u")
        elseif SURFACE
            vtk[VTKPointData()] = ("Scalars" => "phi", "Vectors" => "u")
        else
            vtk[VTKPointData()] = ("Vectors" => "u",)
        end
        pvd[job.t_si] = vtk
    end
    vtk_save(pvd)
    return nothing
end

# Right-handed frame. e1 × e2 = dir. Nothing when the direction vanishes.
function _vtk_axis_frame(dx, dy, dz)
    n = hypot(dx, dy, dz)
    n <= 0 && return nothing
    dx /= n; dy /= n; dz /= n
    ax, ay, az = abs(dz) < 0.9 ? (0.0, 0.0, 1.0) : (1.0, 0.0, 0.0)
    e1x = ay * dz - az * dy
    e1y = az * dx - ax * dz
    e1z = ax * dy - ay * dx
    e1n = hypot(e1x, e1y, e1z)
    e1n <= 0 && return nothing
    e1x /= e1n; e1y /= e1n; e1z /= e1n
    e2x = dy * e1z - dz * e1y
    e2y = dz * e1x - dx * e1z
    e2z = dx * e1y - dy * e1x
    return (dx, dy, dz, e1x, e1y, e1z, e2x, e2y, e2z)
end

# Gas, including an empty flag. A wall or any metal cell closes the envelope.
function _vtk_open_cell(flags, ix, iy, iz, Nx, Ny, Nz)
    (1 <= ix <= Nx && 1 <= iy <= Ny && 1 <= iz <= Nz) || return false
    f = flags[ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny]
    (f & TYPE_S) != 0 && return false
    su = f & TYPE_SU
    return su == TYPE_G || su == 0
end

# Cell-units from an interior point along dir until metal, a wall, or the box.
function _vtk_axis_length(flags, x, y, z, dx, dy, dz, Nx, Ny, Nz)
    lo = 0.5
    tmax = Inf
    for (p, d, hi) in (
        (Float64(x), Float64(dx), Float64(Nx) + 0.5),
        (Float64(y), Float64(dy), Float64(Ny) + 0.5),
        (Float64(z), Float64(dz), Float64(Nz) + 0.5),
    )
        if d > 1.0e-8
            tmax = min(tmax, (hi - p) / d)
        elseif d < -1.0e-8
            tmax = min(tmax, (lo - p) / d)
        end
    end
    (isfinite(tmax) && tmax > 0.05) || return 0.0
    t = 0.0
    step = 0.25
    while t < tmax - 1.0e-6
        tnext = min(tmax, t + step)
        ix = floor(Int, Float64(x) + tnext * Float64(dx) + 0.5)
        iy = floor(Int, Float64(y) + tnext * Float64(dy) + 0.5)
        iz = floor(Int, Float64(z) + tnext * Float64(dz) + 0.5)
        _vtk_open_cell(flags, ix, iy, iz, Nx, Ny, Nz) || return tnext
        t = tnext
    end
    return tmax
end

# Straight cylinder of the given cell radius. Two rings, a side strip, and
# both caps. Coordinates match the rectilinear grid: cell c → (c − 1) Δx.
function _vtk_add_cylinder!(xs, ys, zs, strips, polys,
                            x, y, z, dx, dy, dz, radius, flags,
                            Nx, Ny, Nz, dx_m; nseg::Int=24)
    radius <= 0 && return false
    frame = _vtk_axis_frame(dx, dy, dz)
    frame === nothing && return false
    dirx, diry, dirz, e1x, e1y, e1z, e2x, e2y, e2z = frame
    len = _vtk_axis_length(flags, x, y, z, dirx, diry, dirz, Nx, Ny, Nz)
    len <= 0 && return false
    R = Float64(radius)
    base = length(xs)
    for ring in 0:1
        ox = Float64(x) + ring * len * dirx
        oy = Float64(y) + ring * len * diry
        oz = Float64(z) + ring * len * dirz
        for i in 0:(nseg - 1)
            θ = 2π * i / nseg
            cθ, sθ = cos(θ), sin(θ)
            push!(xs, Float32(ox - 1 + R * (cθ * e1x + sθ * e2x)) * dx_m)
            push!(ys, Float32(oy - 1 + R * (cθ * e1y + sθ * e2y)) * dx_m)
            push!(zs, Float32(oz - 1 + R * (cθ * e1z + sθ * e2z)) * dx_m)
        end
    end
    side = Int[]
    for i in 1:nseg
        push!(side, base + i)
        push!(side, base + nseg + i)
    end
    push!(side, base + 1)
    push!(side, base + nseg + 1)
    push!(strips, MeshCell(PolyData.Strips(), side))
    push!(polys, MeshCell(PolyData.Polys(), [base + i for i in nseg:-1:1]))
    push!(polys, MeshCell(PolyData.Polys(), [base + nseg + i for i in 1:nseg]))
    return true
end

function _vtk_write_poly(dir, stem, t, t_si, xs, ys, zs, groups, data)
    parts = filter(!isempty, groups)
    isempty(parts) && return nothing
    path = joinpath(dir, @sprintf("%s_%08d", stem, t))
    pvd_path = joinpath(dir, stem)
    pvd = paraview_collection(pvd_path; append = t > 0 && isfile(pvd_path * ".pvd"))
    vtk_grid(path, xs, ys, zs, parts...) do vtk
        for (name, vals) in data
            vtk[name] = vals
        end
        pvd[t_si] = vtk
    end
    vtk_save(pvd)
    return nothing
end

# Ray polylines in rays.pvd, and the incident 1/e² cylinder in beam.pvd.
# Point data "power" is watts left on a ray, and the incident power on the cylinder.
function _write_laser_rays(model, domain, dir)
    L = model.laser
    (L === nothing || !L.enabled || L.P <= 0) && return nothing
    rays = trace_laser_rays(model, domain)
    U = model.units
    dx = Float32(U.m)
    isfinite(dx) && dx > 0 || (dx = 1.0f0)
    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    power = Float32[]
    lines = MeshCell{PolyData.Lines, Vector{Int}}[]
    for ray in rays
        ids = Int[]
        for (x, y, z, p) in ray
            push!(xs, (Float32(x) - 1) * dx)
            push!(ys, (Float32(y) - 1) * dx)
            push!(zs, (Float32(z) - 1) * dx)
            push!(power, Float32(p))
            push!(ids, length(xs))
        end
        length(ids) >= 2 && push!(lines, MeshCell(PolyData.Lines(), ids))
    end
    t = Int(domain.t)
    t_si = Float64(si_t(U, t))
    isfinite(t_si) || (t_si = Float64(t))
    _vtk_write_poly(dir, "rays", t, t_si, xs, ys, zs,
                    (lines,), ("power" => power,))
    cxs = Float32[]
    cys = Float32[]
    czs = Float32[]
    strips = MeshCell{PolyData.Strips, Vector{Int}}[]
    polys = MeshCell{PolyData.Polys, Vector{Int}}[]
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    if _vtk_add_cylinder!(cxs, cys, czs, strips, polys,
                          L.x, L.y, L.z, L.dx, L.dy, L.dz, L.w,
                          flags, Nx, Ny, Nz, dx)
        cpower = fill(Float32(L.P), length(cxs))
        _vtk_write_poly(dir, "beam", t, t_si, cxs, cys, czs,
                        (strips, polys), ("power" => cpower,))
    end
    return nothing
end

# One 1/e² cylinder per enabled jet, from the nozzle along its axis to the
# first wall or metal. Open powder.pvd beside lbm.pvd. "mdot" on a tube is
# that nozzle's share, kg/s. A jet that is off draws nothing.
function _write_powder_jet(model, domain, dir)
    jets = _powder_jet_list(model.powder_jet)
    isempty(jets) && return nothing
    U = model.units
    dx = Float32(U.m)
    isfinite(dx) && dx > 0 || (dx = 1.0f0)
    xs = Float32[]
    ys = Float32[]
    zs = Float32[]
    mdot = Float32[]
    strips = MeshCell{PolyData.Strips, Vector{Int}}[]
    polys = MeshCell{PolyData.Polys, Vector{Int}}[]
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    wrote = false
    for J in jets
        (J.enabled && J.w > 0) || continue
        n0 = length(xs)
        _vtk_add_cylinder!(xs, ys, zs, strips, polys,
                           J.x, J.y, J.z, J.dx, J.dy, J.dz, J.w,
                           flags, Nx, Ny, Nz, dx) || continue
        share = Float32(J.mdot)
        resize!(mdot, length(xs))
        mdot[(n0 + 1):end] .= share
        wrote = true
    end
    wrote || return nothing
    t = Int(domain.t)
    t_si = Float64(si_t(U, t))
    isfinite(t_si) || (t_si = Float64(t))
    _vtk_write_poly(dir, "powder", t, t_si, xs, ys, zs,
                    (strips, polys), ("mdot" => mdot,))
    return nothing
end

function _vtk_devbuf(proto, n::Int)
    buf = _VTK_DEV[]
    if buf === nothing || length(buf) < n || typeof(buf) !== typeof(similar(proto, Float32, 1))
        _VTK_DEV[] = similar(proto, Float32, n)
        buf = _VTK_DEV[]
    end
    return buf
end

function _vtk_grow!(slot::_VtkSlot, n::Int, pin::Bool)
    length(slot.host) == n && return slot.host
    slot.host = Vector{Float32}(undef, n)
    pin && CUDA.pin(slot.host)
    return slot.host
end

"""
    flush_exports!()

Wait until queued VTK writes have finished. Called automatically at exit.
"""
function flush_exports!()
    _VTK_STARTED[] || return nothing
    a = take!(_VTK_FREE)
    b = take!(_VTK_FREE)
    put!(_VTK_FREE, a)
    put!(_VTK_FREE, b)
    return nothing
end

function export!(model::Model; dir::AbstractString="output", fields=nothing, sync::Bool=false)
    start_run_log!(dir)
    model.initialized || initialize!(model)
    moments!(model)
    mkpath(dir)

    domain = model.domains[1]
    _write_laser_rays(model, domain, dir)
    _write_powder_jet(model, domain, dir)
    t = Int(domain.t)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    N = Nx * Ny * Nz
    U = model.units
    mask = _vtk_mask(fields)
    offs, ncomp = _vtk_offsets(mask)
    ncomp == 0 && return nothing
    ρs = Float32(U.kg / U.m^3)
    us = Float32(U.m / U.s)
    ps = Float32(ρs * us * us / 3)
    Ts = Float32(U.K)
    Ss = Float32(ρs / U.s)
    Qs = Float32(ρs * U.cp * U.K / U.s)
    Tdata = @static TEMPERATURE ? domain.T.data : domain.ρ.data
    fsdata = @static TEMPERATURE ? domain.fs.data : domain.ρ.data
    Qdata = @static TEMPERATURE ? domain.Q.data : domain.ρ.data
    ϕdata = @static SURFACE ? domain.ϕ.data : domain.ρ.data
    mpdata = @static SURFACE ? domain.mp.data : domain.ρ.data
    Sdata = @static SURFACE ? domain.msrc.data : domain.ρ.data

    dev = _vtk_devbuf(domain.ρ.data, N * ncomp)
    pack_vtk_kernel!(model.backend, model.workgroup)(
        dev, domain.ρ.data, domain.u.data, domain.flags.data,
        Tdata, fsdata, ϕdata, mpdata, Sdata, Qdata,
        N, offs.rho, offs.p, offs.u, offs.T, offs.fs, offs.phi, offs.mp, offs.S, offs.Q, offs.flags,
        ρs, ps, us, Ts, Ss, Qs; ndrange=N)

    dx = Float32(U.m)
    isfinite(dx) && dx > 0 || (dx = 1.0f0)
    t_si = Float64(si_t(U, t))
    isfinite(t_si) || (t_si = Float64(t))
    # The write runs on a second Julia thread. With one thread it would never
    # start, and the next export would block forever waiting for a free slot.
    use_async = !sync && model.backend isa CUDABackend && CUDA.functional() && Threads.nthreads() >= 2

    if !use_async
        host = Array{Float32}(undef, N * ncomp)
        copyto!(host, @view(dev[1:(N * ncomp)]))
        _vtk_write(_VtkJob(dir, t, t_si, dx, Nx, Ny, Nz, host, offs, 0, nothing, nothing))
        return nothing
    end

    _ensure_vtk_writer()
    prev = _VTK_COPY_EV[]
    if prev !== nothing
        CUDA.synchronize(prev)
    end
    slot_i = take!(_VTK_FREE)
    host = _vtk_grow!(_VTK_SLOTS[slot_i], N * ncomp, true)
    if _VTK_STREAM[] === nothing
        _VTK_STREAM[] = CUDA.CuStream()
    end
    packed = CUDA.CuEvent()
    CUDA.record(packed)
    cs = _VTK_STREAM[]
    CUDA.wait(packed, cs)
    GC.@preserve host dev begin
        unsafe_copyto!(pointer(host), pointer(dev), N * ncomp; stream=cs, async=true)
    end
    done = CUDA.CuEvent()
    CUDA.record(done, cs)
    _VTK_COPY_EV[] = done
    put!(_VTK_JOBS, _VtkJob(dir, t, t_si, dx, Nx, Ny, Nz, host, offs, slot_i, done, CUDA.context()))
    return nothing
end