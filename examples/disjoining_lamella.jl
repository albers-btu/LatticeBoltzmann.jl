# LBfoam Fig. 4 analog: two bubbles grow toward each other in a supersaturated
# liquid. k_Π = 0.08 holds a film; set k_Π = 0 to watch them merge on contact.
#
# Over-pressure alone is not enough: p = n/V falls as they swell and they stop
# before the gap closes. Dissolved gas (c∞ > k_H p) keeps feeding n so they
# actually meet.
#
# ParaView: output_disjoining_lamella/lbm.pvd. Contour phi = 0.5 (Threshold
# flags 8–32). Two flattened caps with a sheet between = lamella; one blob =
# coalesced. Watch `gap` in the log — it should drop from ~6 toward ~1–2.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_disjoining_lamella")

Nx, Ny, Nz = 56, 28, 28
R0 = 6.0f0
# Centres 18 apart, R=6 → liquid gap ≈ 6 cells (inside d_max=4 after a little growth).
c1 = (18.0f0, Float32(Ny + 1) / 2, Float32(Nz + 1) / 2)
c2 = (36.0f0, c1[2], c1[3])
k_Π = 0.08f0                 # 0 → coalesce
σ = 0.008f0
k_H = 3.0f0
α_c = 0.2f0
c∞ = 1.45f0                  # supersaturated; keeps n rising
n_over = 1.25f0
nsteps = 350
every = 10

backend = CUDA.functional() ? CUDABackend() : CPU()
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=σ, T_avg=1,
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; k_Π=k_Π, d_max=4))

host = zeros(UInt8, Nx * Ny * Nz)
ch = fill(c∞, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
        ch[n] = k_H * P_ATM_LAT
        continue
    end
    d1 = (Float32(x) - c1[1])^2 + (Float32(y) - c1[2])^2 + (Float32(z) - c1[3])^2
    d2 = (Float32(x) - c2[1])^2 + (Float32(y) - c2[2])^2 + (Float32(z) - c2[3])^2
    if d1 <= R0 * R0 || d2 <= R0 * R0
        host[n] = TYPE_G
        ch[n] = k_H * P_ATM_LAT
    else
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)
for r in bubble_records(model)
    set_bubble_n!(model, r.id, n_over * P_ATM_LAT * r.V)
end
export!(model; dir="output_disjoining_lamella")

function film_gap(model)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    fl = Array(d.flags.data)
    bd = Array(d.bid.data)
    xmax1 = 0
    xmin2 = Nx + 1
    yc, zc = (Ny + 1) ÷ 2, (Nz + 1) ÷ 2
    for z in (zc - 3):(zc + 3), y in (yc - 3):(yc + 3), x in 2:(Nx - 1)
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        su = fl[n] & TYPE_SU
        (su == TYPE_G || su == TYPE_I) || continue
        id = Int(round(bd[n]))
        id == 1 && (xmax1 = max(xmax1, x))
        id == 2 && (xmin2 = min(xmin2, x))
    end
    return xmin2 - xmax1 - 1
end

gap0 = film_gap(model)
@info "disjoining lamella" Nx Ny Nz k_Π σ c∞ gap0 n_over nsteps backend

function run_lamella!(model, nsteps, every)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="lamella ")
    t = 0
    local recs = bubble_records(model)
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_disjoining_lamella")
        recs = bubble_records(model)
        gap = film_gap(model)
        next!(prog; showvalues = [
            (:t, t),
            (:nb, length(recs)),
            (:gap, gap),
            (:R, isempty(recs) ? 0.0 : round(sum(r -> r.R, recs) / length(recs); digits=2)),
            (:V, isempty(recs) ? 0.0 : round(sum(r -> r.V, recs); digits=1)),
        ])
    end
    finish!(prog)
    return recs
end

recs = run_lamella!(model, nsteps, every)
@info "disjoining lamella done" nb=length(recs) gap=film_gap(model) recs
