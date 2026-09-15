# Dissolved-gas bubble growth (LBfoam / Epstein–Plesset). A 316L-like liquid
# box holds one enclosed bubble. Supersaturated c∞ > k_H p drives Henry flux
# into the cavity; pV = nT then inflates it. Set c∞ below k_H p_atm to dissolve.
#
# ParaView: open output_bubble_growth/lbm.pvd, colour by c (or T), Contour
# phi = 0.5 for the free surface. pgas is the bubble pressure.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using Printf
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_bubble_growth")

Nx = Ny = Nz = 32
xc = yc = zc = Float32(Nx + 1) / 2
R0 = 6.0f0
k_H = 3.0f0                  # c_s = k_H p; atm → c_s = 1
α_c = 0.2f0                  # Model-α = 2D so D = 0.1
c∞ = 1.45f0                  # > 1 → growth; 0.55 → dissolution
σ = 0.0f0
nsteps = 400
every = 20

backend = CUDA.functional() ? CUDABackend() : CPU()
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=σ,
              T_avg=1.0f0, backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}())

host = zeros(UInt8, Nx * Ny * Nz)
ch = fill(c∞, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
        ch[n] = k_H * P_ATM_LAT
    else
        dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
        host[n] = dx * dx + dy * dy + dz * dz <= R0 * R0 ? TYPE_G : TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)
export!(model; dir="output_bubble_growth")
b0 = bubble_records(model)[1]
D = dissolved_D(model.domains[1])
@info "bubble growth" Nx Ny Nz R0=b0.R n0=b0.n p0=b0.p c∞ k_H D α_c nsteps backend

function run_growth!(model, b0, D, c∞, k_H, nsteps, every)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="bubble growth ")
    t = 0
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_bubble_growth")
        recs = bubble_records(model)
        b = isempty(recs) ? nothing : recs[1]
        R2an = b === nothing ? NaN : epstein_plesset_R2(b0.R, t, D, c∞, k_H, P_ATM_LAT)
        next!(prog; showvalues = [
            (:t, t),
            (:nb, b === nothing ? 0 : 1),
            (:R, b === nothing ? NaN : round(b.R; digits=3)),
            (:R_EP, b === nothing ? NaN : round(sqrt(max(R2an, 0)); digits=3)),
            (:n, b === nothing ? NaN : round(b.n; digits=2)),
            (:p, b === nothing ? NaN : round(b.p; digits=4)),
        ])
    end
    finish!(prog)
    return bubble_records(model)
end

recs = run_growth!(model, b0, D, c∞, k_H, nsteps, every)
@info "bubble growth done" recs
