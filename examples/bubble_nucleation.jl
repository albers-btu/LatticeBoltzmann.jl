# Homogeneous nucleation in a supersaturated liquid (LBfoam Poisson-disk).
# No seed bubble: nuclei appear where c > c_star, at least d_min apart, then
# grow by Henry + pV = nT.
#
# ParaView: output_bubble_nucleation/lbm.pvd. Colour by c, Contour phi = 0.5
# (or Threshold flags 8–32 first). Each blob is a nucleus/bubble.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_bubble_nucleation")

Nx = Ny = Nz = 40
k_H = 3.0f0
α_c = 0.2f0
c∞ = 1.45f0
c_star = 1.10f0
d_min = 8.0f0
nsteps = 400
every = 20

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=d_min, R=1, c_star=c_star, p_cell=1,
                          n_max=6, n_over=1.3f0, every=25, n_total_max=40)
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0,
              T_avg=1.0f0, backend=backend,
              workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc))

host = zeros(UInt8, Nx * Ny * Nz)
ch = fill(c∞, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
        ch[n] = k_H * P_ATM_LAT
    else
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)
export!(model; dir="output_bubble_nucleation")
D = dissolved_D(model.domains[1])
@info "bubble nucleation" Nx Ny Nz c∞ c_star d_min k_H D nsteps backend

function run_nucleation!(model, nsteps, every)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="nucleation ")
    t = 0
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_bubble_nucleation")
        recs = bubble_records(model)
        npl = model.bubbles.nucleation.n_planted
        next!(prog; showvalues = [
            (:t, t),
            (:planted, npl),
            (:nb, length(recs)),
            (:Rmax, isempty(recs) ? 0.0 : round(maximum(r -> r.R, recs); digits=3)),
            (:n_sum, isempty(recs) ? 0.0 : round(sum(r -> r.n, recs); digits=1)),
        ])
    end
    finish!(prog)
    return bubble_records(model)
end

recs = run_nucleation!(model, nsteps, every)
@info "bubble nucleation done" n_planted=model.bubbles.nucleation.n_planted recs
