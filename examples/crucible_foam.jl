# LBfoam-style bucket: liquid in a crucible, free surface, blowing agent,
# dissolved gas, nucleation, growth, then a freeze so pores stay in the mush.
# No laser.
#
# ParaView: output_crucible_foam/lbm.pvd. Colour by a or c, Contour phi = 0.5
# (Threshold flags 8–32). After the cool-down, pores sit in frozen metal (fs).
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_crucible_foam")

Nx = Ny = 32
Nz = 40
Hfill = 22
k_H = 3.0f0
α_c = 0.2f0
a0 = 0.07f0
k_a = 0.03f0
E_a = 0.0f0
T_m = 1.0f0
T_hot = 1.2f0
n_foam = 240
n_cool = 80
every = 20

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=7, R=1, c_star=1.12f0, p_cell=1,
                          n_max=5, n_over=1.3f0, every=20, n_total_max=40)
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0,
              k_a=k_a, E_a=E_a, Y_a=1, a_fs_max=0.99, T_avg=T_hot,
              Λ=0.5f0, Ts=T_m, Tl=T_m, K0=1.0f-3,
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc))

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_hot, Nx * Ny * Nz)
ah = zeros(Float32, Nx * Ny * Nz)
ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
Qh = zeros(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
        Th[n] = T_hot
    elseif z <= Hfill
        host[n] = TYPE_F
        ah[n] = a0
    else
        host[n] = TYPE_G
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].a.data, ah)
copyto!(model.domains[1].c.data, ch)
copyto!(model.domains[1].Q.data, Qh)

LatticeBoltzmann.initialize!(model)
export!(model; dir="output_crucible_foam")
m0 = foam_metrics(model)
@info "crucible foam" Nx Ny Nz Hfill a0 k_a T_hot n_foam n_cool m0 backend

function run_crucible!(model, nsteps, every; tag="foam")
    prog = Progress(cld(nsteps, every); dt=0.3, desc="$tag ")
    t = 0
    local m = foam_metrics(model)
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_crucible_foam")
        m = foam_metrics(model)
        next!(prog; showvalues = [
            (:t, t),
            (:nb, m.nb),
            (:porosity, round(m.porosity; digits=3)),
            (:fill_z, round(m.fill_z; digits=2)),
            (:a, round(m.a; digits=3)),
            (:frozen, m.n_frozen),
        ])
    end
    finish!(prog)
    return m
end

m1 = run_crucible!(model, n_foam, every; tag="foam")
@info "foam phase done" m1

# Quench: lock the mush so pores cannot refill.
fsA = Array(model.domains[1].fs.data)
fl = Array(model.domains[1].flags.data)
for n in eachindex(fl)
    su = fl[n] & TYPE_SU
    (su == TYPE_F || su == TYPE_I) && (fsA[n] = 1.0f0)
end
copyto!(model.domains[1].fs.data, fsA)
fill!(model.domains[1].Q.data, 0)
m2 = run_crucible!(model, n_cool, every; tag="freeze")
@info "crucible foam done" m0 m1 m2
