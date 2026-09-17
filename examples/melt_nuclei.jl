# Isolated melt nucleation: a liquid pad with a free surface, no laser.
# The coupled laser_foam case cannot plant a 3³ embryo in a 6-cell keyhole
# film without the ray-traced beam dumping into the crater (FSLBM Ma→1).
# This example is that physics without the beam: supersaturated liquid,
# Poisson-disk nuclei under the lid, Henry + pV=nT growth, disjoining films.
#
# ParaView: output_melt_nuclei/lbm.pvd. Threshold flags 8–32, Contour phi=0.5.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_melt_nuclei")

Nx = Ny = 48
Nz = 40
Hfill = 20
k_H = 3.0f0
c∞ = 1.40f0
c_star = 1.10f0
σ = 0.02f0
nsteps = 80
every = 10

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=8, R=1, c_star=c_star, p_cell=1,
                          n_max=3, n_over=1.0f0, every=10, n_total_max=8)
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0.2f0, k_H=k_H, fz=0, σ=σ,
              T_avg=1.05f0, Λ=0.6f0, Ts=1.0f0, Tl=1.0f0, K0=1.0f-3,
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(1.05f0, Nx * Ny * Nz)
fsh = zeros(Float32, Nx * Ny * Nz)
ch = fill(c∞, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
        fsh[n] = 1
        ch[n] = k_H * P_ATM_LAT
    elseif z <= Hfill
        host[n] = TYPE_F
        fsh[n] = 0
    else
        host[n] = TYPE_G
        fsh[n] = 0
        ch[n] = k_H * P_ATM_LAT
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)
export!(model; dir="output_melt_nuclei")
m0 = foam_metrics(model)
@info "melt nuclei" Nx Ny Nz Hfill c∞ c_star σ nsteps m0 backend

function metal_umax(model)
    d = model.domains[1]
    uA = Array(d.u.data)
    fl = Array(d.flags.data)
    um = 0.0f0
    @inbounds for n in axes(uA, 1)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        um = max(um, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    return um
end

function bubble_z_mean(model)
    B = model.bubbles
    B isa BubbleTracker || return 0.0
    Nx, Ny = Int(model.domains[1].Nx), Int(model.domains[1].Ny)
    s = 0.0
    n = 0
    @inbounds for i in eachindex(B.label)
        B.label[i] > 0 || continue
        s += (i - 1) ÷ (Nx * Ny) + 1
        n += 1
    end
    return n == 0 ? 0.0 : s / n
end

function run_melt_nuclei!(model, nsteps, every)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="melt nuclei ")
    t = 0
    m = foam_metrics(model)
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_melt_nuclei")
        m = foam_metrics(model)
        umax = metal_umax(model)
        next!(prog; showvalues = [
            (:t, t),
            (:nb, m.nb),
            (:planted, m.n_planted),
            (:z_bub, round(bubble_z_mean(model); digits=1)),
            (:porosity, round(m.porosity; digits=3)),
            (:umax, round(umax; digits=3)),
            (:nF, m.nF),
        ])
    end
    finish!(prog)
    return m
end

m = run_melt_nuclei!(model, nsteps, every)
umax = metal_umax(model)
@info "melt nuclei done" m0 m umax z_bub=bubble_z_mean(model)
