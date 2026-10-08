# Feature-path step time. MLUPS is get_N / wall time of one outer step!,
# the same definition run! prints. It is not a pass/fail threshold.
#
# Metal mass may rise by at most the powder injected during the timed steps,
# plus 25% and one cell. It may fall by at most 5% of the mass after warmup
# plus one cell. T stays finite and ≥ 0. |u| stays below 0.9. Populations
# and mass stay finite.
#
#   julia -t2 --project=. examples/gpu_feature_bench.jl
#   julia -t2 --project=. examples/gpu_feature_bench.jl sweep
#
# sweep times workgroups 128, 256, and 512 on the small feature grid.
# The default stays 256 unless a later change records a win here.
# Export frames go to /tmp/lbm_gpu_feature_export, not a track directory.

using LatticeBoltzmann
using CUDA
using KernelAbstractions
using Printf

function median(xs)
    ys = sort(xs)
    n = length(ys)
    n == 0 && return NaN
    return isodd(n) ? ys[(n + 1) ÷ 2] : 0.5 * (ys[n ÷ 2] + ys[n ÷ 2 + 1])
end

function paint_plate!(model, H)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    host = fill(TYPE_G, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z <= H
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    return nothing
end

function place_jets!(model, H)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    U = model.units
    cx, cy, zf = 0.5 * (Nx + 1), 0.5 * (Ny + 1), Float64(H) + 0.5
    height = min(20.0, Nz - zf - 2)
    radius = height * tan(π / 6)
    jets = PowderJet[]
    for θdeg in (90, 210, 330)
        θ = deg2rad(θdeg)
        J = PowderJet(U; mdot=1.0e-4, w=2.0, v=1.0, d=0.4,
                      x=cx + radius * cos(θ), y=cy + radius * sin(θ), z=zf + height,
                      nparcels=8, nmax=32, enabled=true)
        aim_powder_jet!(J, cx, cy, zf)
        push!(jets, J)
    end
    model.powder_jet = jets
    return jets
end

function build_model(Nx, Ny, Nz; n_hydro, every, laser_on, powder_on, workgroup, backend)
    # Lattice constructor. Evaporation is Λ_v, C_hk, p0v; radiation is C_rad;
    # loose-powder lifetime is τ_p. The Units constructor names differ.
    model = Model(Nx, Ny, Nz, 0.05f0;
                  α=0.02f0, σ=0.01f0, σT=-1.0f-4, fz=-1.0f-5,
                  Λ=0.05f0, Ts=1.0f0, Tl=1.0f0,
                  Λ_v=0.2f0, T_v=1.5f0, β_v=1.0f0,
                  C_hk=1.0f-3, p0v=1.0f-3,
                  C_rad=1.0f-4, T_rad=1.0f0,
                  τ_p=50.0f0, n_hydro=n_hydro,
                  backend=backend, workgroup=workgroup)
    H = max(2, Nz ÷ 6)
    paint_plate!(model, H)
    if laser_on
        U = model.units
        model.laser = Laser(U; P=0.01, w=3.0, x=0.5 * (Nx + 1), y=0.5 * (Ny + 1),
                             z=Nz - 1.5, nrays=11, max_bounce=2, every=every, skin=2)
    end
    powder_on && place_jets!(model, H)
    initialize!(model)
    return model
end

function injected_mass(model, nsteps)
    jets = LatticeBoltzmann._powder_jet_list(model.powder_jet)
    U = model.units
    s = 0.0
    for J in jets
        J.enabled || continue
        s += Float64(J.mdot) * Float64(U.s) / Float64(U.kg) * nsteps
    end
    return s
end

function metal_mass(model)
    d = model.domains[1]
    mass = Array(d.mass.data)
    flags = Array(d.flags.data)
    s = 0.0
    @inbounds for i in eachindex(mass)
        su = flags[i] & TYPE_SU
        if su == TYPE_F || su == TYPE_I || su == TYPE_IF
            s += Float64(mass[i])
        end
    end
    return s
end

function check_guards(model, m0, injected)
    d = model.domains[1]
    moments!(model)
    T = Array(d.T.data)
    u = Array(d.u.data)
    fi = Array(d.fi.data)
    mass = Array(d.mass.data)
    finite_T = all(isfinite, T)
    Tmin = finite_T ? minimum(T) : NaN
    Tmax = finite_T ? maximum(T) : NaN
    umax = 0.0
    @inbounds for i in 1:size(u, 1)
        umax = max(umax, hypot(Float64(u[i, 1]), Float64(u[i, 2]), Float64(u[i, 3])))
    end
    m1 = metal_mass(model)
    ok = finite_T && Tmin >= 0 && umax < 0.9 && all(isfinite, fi) && all(isfinite, mass)
    ok = ok && m1 <= m0 + injected * 1.25 + 1.0
    ok = ok && m1 >= m0 - 0.05 * abs(m0) - 1.0
    return ok, Tmin, Tmax, umax, m1
end

function report_run(tag, model, ts, m0, ntimed)
    ms = median(ts)
    mlups = get_N(model) * 1e-3 / ms
    injected = injected_mass(model, ntimed)
    ok, Tmin, Tmax, umax, m1 = check_guards(model, m0, injected)
    L = model.laser
    nrays = L === nothing ? 0 : length(L.Pray)
    every = L === nothing ? 0 : L.every
    @printf("%s  N=%d  n_hydro=%d  rays=%d  every=%d  workgroup=%d\n",
            tag, get_N(model), model.n_hydro, nrays, every, model.workgroup)
    @printf("  clock-off median %.3f ms/step   %.2f MLUPS\n", ms, mlups)
    @printf("  T=[%.4g, %.4g]  |u|max=%.4g  metal %.6g → %.6g  injected %.6g  %s\n",
            Tmin, Tmax, umax, m0, m1, injected, ok ? "ok" : "GUARD FAILED")
    return ok
end

function run_clock(model, nwarm, ntimed)
    for _ in 1:nwarm
        LatticeBoltzmann.step!(model)
    end
    m0 = metal_mass(model)
    ts = Float64[]
    for _ in 1:ntimed
        t0 = time_ns()
        LatticeBoltzmann.step!(model)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return ts, m0
end

function run_phases(model, n)
    model.phase_ns = zeros(6)
    acc = zeros(6, n)
    for j in 1:n
        fill!(model.phase_ns, 0.0)
        LatticeBoltzmann.step!(model)
        acc[:, j] .= model.phase_ns ./ 1e6
    end
    model.phase_ns = nothing
    names = ("host+laser+powder", "powder-gas", "surface_0", "moving", "collide", "surface_1/2/3")
    println("  clock-on phase median ms (attribution, slower than the gate):")
    for i in 1:6
        @printf("    %-18s %8.3f\n", names[i], median(acc[i, :]))
    end
    return nothing
end

function cpu_checks()
    model = build_model(16, 16, 16; n_hydro=2, every=1, laser_on=true, powder_on=true,
                         workgroup=32, backend=CPU())
    LatticeBoltzmann.step!(model)
    d = model.domains[1]
    qsum = sum(Array(d.Q.data))
    (isfinite(qsum) && qsum > 0) || error("CPU laser deposit sum=$(qsum)")
    dir = mktempdir()
    export!(model; dir)
    ray = read(joinpath(dir, "rays_00000001.vtp"), String)
    occursin("Lines", ray) || error("rays.pvd is not lines")
    occursin("Strips", ray) && error("rays.pvd contains a strip")
    beam = read(joinpath(dir, "beam_00000001.vtp"), String)
    occursin("NumberOfPoints=\"48\"", beam) || error("beam cylinder is not 48 points")
    pow = read(joinpath(dir, "powder_00000001.vtp"), String)
    occursin("mdot", pow) || error("powder cylinder has no mdot")
    rm(dir; recursive=true)
    J = model.powder_jet[1]
    for P in model.powder_jet
        P.enabled = false
        P.alive .= false
        P.qhold_dirty = false
    end
    J.qhold_dirty = true
    fill!(J.qhold, 0)
    J.qhold[1] = 0.25f0
    fill!(d.Q.data, 0)
    advance_powder_jet!(model, d, false)
    q1 = Array(d.Q.data)[1]
    abs(Float64(q1) - 0.25) < 1.0e-6 || error("CPU qhold restore wrote $(q1)")
    J.qhold_dirty == false || error("qhold_dirty stayed set")
    qbefore = copy(Array(d.Q.data))
    advance_powder_jet!(model, d, false)
    qbefore == Array(d.Q.data) || error("idle powder step changed Q")
    println("CPU deposit, idle qhold, and export meshes ok")
    return nothing
end

cpu_checks()

if any(==("cpu"), ARGS)
    println("CPU checks only")
    exit(0)
end

if !CUDA.functional()
    println("CUDA is not functional. No GPU speedup is claimed.")
    exit(0)
end

const BACKEND = CUDABackend()
dev = CUDA.device()
println("device: ", CUDA.name(dev))
println("threads: ", Threads.nthreads(),
        "   async export: ", Threads.nthreads() >= 2)

function one(tag, Nx, Ny, Nz; n_hydro, every, laser_on, powder_on, nwarm, ntimed, phases=false, workgroup=256)
    println("starting $tag")
    model = build_model(Nx, Ny, Nz; n_hydro, every, laser_on, powder_on,
                         workgroup, backend=BACKEND)
    ts, m0 = run_clock(model, nwarm, ntimed)
    ok = report_run(tag, model, ts, m0, ntimed)
    phases && run_phases(model, 4)
    return ok, model
end

ok = true
good, small = one("feature-every1", 64, 32, 48; n_hydro=15, every=1, laser_on=true, powder_on=true,
                  nwarm=8, ntimed=20, phases=true)
ok &= good
good, _ = one("feature-every4", 64, 32, 48; n_hydro=15, every=4, laser_on=true, powder_on=true,
              nwarm=8, ntimed=20)
ok &= good
good, _ = one("hydro-only", 64, 32, 48; n_hydro=1, every=1, laser_on=false, powder_on=false,
              nwarm=8, ntimed=20)
ok &= good
good, _ = one("tim-shaped", 160, 80, 96; n_hydro=15, every=1, laser_on=true, powder_on=true,
              nwarm=2, ntimed=5)
ok &= good

if any(==("sweep"), ARGS)
    println("workgroup sweep, small feature grid, clock off")
    for wg in (128, 256, 512)
        good, _ = one("sweep-$wg", 64, 32, 48; n_hydro=15, every=1, laser_on=true, powder_on=true,
                      nwarm=4, ntimed=8, workgroup=wg)
        global ok
        ok &= good
    end
end

edir = "/tmp/lbm_gpu_feature_export"
rm(edir; force=true, recursive=true)
t0 = time_ns()
export!(small; dir=edir)
frame_ms = (time_ns() - t0) / 1e6
flush_exports!()
async_taken = small.backend isa CUDABackend && CUDA.functional() && Threads.nthreads() >= 2
@printf("export! returned in %.3f ms   async=%s\n", frame_ms, async_taken)
isfile(joinpath(edir, "rays.pvd")) || error("export did not write rays.pvd")
isfile(joinpath(edir, "beam.pvd")) || error("export did not write beam.pvd")
isfile(joinpath(edir, "powder.pvd")) || error("export did not write powder.pvd")
ray = read(joinpath(edir, filter(f -> startswith(f, "rays_") && endswith(f, ".vtp"), readdir(edir))[1]), String)
occursin("Lines", ray) || error("GPU export rays are not lines")
!occursin("Strips", ray) || error("GPU export rays contain a strip")
println("export frame ok  ", edir)
ok || error("a guard failed")
println("bench done")
