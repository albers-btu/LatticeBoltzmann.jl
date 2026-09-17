# LBfoam Fig. 15 driver (Ataei et al., 2020): 300 × 200 × 200, 40 Poisson-disk
# spheres, 30 000 steps. D3Q7 Henry (eqs. 24–29) and PLIC-ray Π.
#
# k_H = 3 so c = 1 at atmosphere (their 10⁻⁵ is a different c scaling).
# D matches τ_g = 0.53. Paint R = 5 so the G core after the I shell is ~ R = 3.
#
# ParaView: output_lbfoam_fig15/lbm.pvd. Threshold flags 8–32, Contour phi=0.5.
#
#   julia --project=. examples/lbfoam_fig15.jl
#   julia --project=. examples/lbfoam_fig15.jl 50   # short smoke
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Random
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE

Nx, Ny, Nz = 300, 200, 200
Hfill = 80                      # extra liquid above the seeds so they do not vent
n_nuclei = 40
# Paper paints R = 3. FSLBM turns the outer shell into TYPE_I, so a painted
# R = 3 leaves a ~1.7-cell G core that CSF blasts. R = 5 → G core ~ paper R.
R0 = 5.0f0
d_min = 16.0f0                # initial gap > d_max so Π is off until they grow
nsteps_default = 30_000
vtk_times_default = (0, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000)

# Paper: τ = 0.95 → ν = c_s² (τ − 1/2) = 0.15. γ = 5×10⁻³, k_Π = 5×10⁻³.
ν = 0.15f0
σ = 0.005f0
k_Π = 0.005f0                 # Fig. 15 caption
d_max = 4.0f0
k_H = 3.0f0
# Paper τ_g = 0.53, D3Q7 c_s² = 1/4 → D = 0.0075. Here D = α_c / 2.
α_c = 0.015f0
c0 = 1.20f0                   # c_s = k_H / 3 = 1; 1.5 overfills n on step 1
T_gas = 1.0f0
seed = 15

nsteps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : nsteps_default
vtk_times = nsteps >= nsteps_default ? vtk_times_default :
    Tuple(unique(sort(vcat(0, nsteps))))

backend = CUDA.functional() ? CUDABackend() : CPU()

function foam_brief(model)
    recs = bubble_records(model)
    V = 0.0
    Rmax = 0.0
    @inbounds for r in recs
        V += Float64(r.V)
        Rmax = max(Rmax, Float64(r.R))
    end
    return (nb=length(recs), V=V, Rmax=Rmax)
end

function run_fig15!(model, nsteps, vtk_times, dir)
    export!(model; dir)
    prog = Progress(nsteps; dt=0.5, desc="Fig. 15 ")
    brief = foam_brief(model)
    targets = sort(unique(Int[vtk_times..., nsteps]))
    t = Int(model.domains[1].t)
    done = 0
    for t_end in targets
        t_end <= t && continue
        while t < t_end
            nrun = min(50, t_end - t)
            with_logger(NullLogger()) do
                run!(model, nrun)
            end
            t = Int(model.domains[1].t)
            done += nrun
            brief = foam_brief(model)
            next!(prog; step=nrun, showvalues = [
                (:t, t),
                (:nb, brief.nb),
                (:Rmax, round(brief.Rmax; digits=2)),
                (:V_gas, round(brief.V; digits=0)),
            ])
        end
        export!(model; dir)
    end
    finish!(prog)
    return foam_brief(model)
end

function build_fig15()
    Random.seed!(seed)
    pad = 2 + ceil(Int, R0)
    centers = poisson_disk_3d(
        n_nuclei, d_min,
        (1 + pad, Nx - pad),
        (1 + pad, Ny - pad),
        (1 + pad, min(Hfill - pad - 8, 40)),
    )
    model = Model(Nx, Ny, Nz, ν;
                  α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=σ, T_avg=1.0f0,
                  backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
                  bubbles=BubbleTracker{Float32}(;
                      T_gas=T_gas, k_Π=k_Π, d_max=d_max, every=10,
                      nucleation=nothing))
    host = zeros(UInt8, Nx * Ny * Nz)
    paint_spherical_nuclei!(host, Nx, Ny, Nz, centers, R0; Hfill)
    ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
    @inbounds for n in eachindex(host)
        (host[n] & TYPE_SU) == TYPE_F && (ch[n] = c0)
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].c.data, ch)
    LatticeBoltzmann.initialize!(model)
    equilibrate_nuclei_n!(model; n_over=1)
    return model, centers
end

function main()
    start_run_log!("output_lbfoam_fig15")
    model, _ = build_fig15()
    nb0 = model.bubbles.nb
    D = dissolved_D(model.domains[1])
    @info "LBfoam Fig. 15" Nx Ny Nz Hfill n_nuclei R0 d_min nsteps ν σ k_Π k_H α_c D c0 T_gas nb0 backend
    nb0 == n_nuclei || @warn "enclosed bubbles ≠ seeds (merged or vented to atmosphere)" nb0 n_nuclei
    brief = run_fig15!(model, nsteps, vtk_times, "output_lbfoam_fig15")
    @info "LBfoam Fig. 15 done" brief.nb brief.Rmax brief.V nsteps t=Int(model.domains[1].t)
    return model, brief
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
