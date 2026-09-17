# Step 3 of laser–foam coupling: melt a thick pool, beam on, no nucleation.
# Starts solid. The beam must open ≥ 12 liquid TYPE_F cells on the axis
# (room for a 3³ embryo) without eating the pad. Foam is off.
#
# Tests: test/test_laser_pool.jl
# ParaView: output_laser_pool/lbm.pvd. Colour **fs** (1 = solid, 0 = liquid)
# and T 1400–2800 K. Threshold flags 8–32, Contour phi=0.5.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_laser_pool")

si_Lx = 3.0e-3u"m"
si_Ly = 3.0e-3u"m"
si_H  = 1.2e-3u"m"
si_gas = 0.8e-3u"m"
si_dx = 50.0e-6u"m"

si_ρ  = 8000u"kg/m^3"
si_cp = 500.0u"J/kg/K"
si_k_s = 15.0u"W/m/K"
si_k_l = 30.0u"W/m/K"
si_Tm = 1673.0u"K"
si_T_init = 0.88 * si_Tm          # solid; melt front is visible in fs
si_Lheat = 2.7e5u"J/kg"
si_Lv = 7.45e6u"J/kg"
si_Tv = 3086.0u"K"
si_M = 0.0558u"kg/mol"
si_σ = 1.5u"N/m"
si_P = 280.0u"W"
si_d_spot = 0.5e-3u"m"
si_δ = 0.60e-3u"m"                 # nskin ~ 12 cells so the column can melt
si_t_heat = 4.0e-3u"s"

dx = ustrip(u"m", si_dx)
L = max(4, round(Int, ustrip(u"m", si_H) / dx))
m = ustrip(u"m", si_H) / L
Nx = max(16, round(Int, ustrip(u"m", si_Lx) / m))
Ny = max(16, round(Int, ustrip(u"m", si_Ly) / m))
Hfill = L + 2
n_gas_top = max(6, ceil(Int, ustrip(u"m", si_gas) / m))
Nz = Hfill + n_gas_top + 1

α_s_si = si_k_s / (si_ρ * si_cp)
α_l_si = si_k_l / (si_ρ * si_cp)
si_ν = ustrip(u"m^2/s", α_l_si) * u"m^2/s"
units = Units(si_H, si_ρ, si_σ; x=L, σ_lat=0.03, u=0.05, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)
s = units.s
n_heat = max(40, round(Int, ustrip(u"s", si_t_heat) / s))
every = max(10, n_heat ÷ 8)

T_init = Float32(lbm_T(units, si_T_init))
nskin = max(3, round(Int, ustrip(u"m", si_δ) / m))
A0 = fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
backend = CUDA.functional() ? CUDABackend() : CPU()

model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_l_si, α_s=2 * α_s_si, α_l=2 * α_l_si,
              ν_s=0.5 * si_ν, ν_l=si_ν,
              gz=0, σ=si_σ,
              T_avg=T_init, latent=si_Lheat, Ts=si_Tm, Tl=si_Tm,
              latent_v=si_Lv, T_v=si_Tv, M=si_M,
              emissivity=0.4, T_rad=si_T_init,
              K0=1.0e-10u"m^2",
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64)

xc = (Nx + 1) / 2
yc = (Ny + 1) / 2
model.laser = Laser(units; P=si_P, w=0.5 * si_d_spot,
                    x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                    nrays=11, max_bounce=4, every=1, skin=nskin)

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_init, Nx * Ny * Nz)
fsh = ones(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S | TYPE_T
        fsh[n] = 1
        Th[n] = T_init
    elseif z <= Hfill
        host[n] = TYPE_F
        fsh[n] = 1
        Th[n] = T_init
    else
        host[n] = TYPE_G
        fsh[n] = 0
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
LatticeBoltzmann.initialize!(model)

ix = round(Int, xc)
iy = round(Int, yc)

function pool_probe(model)
    d = model.domains[1]
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    uA = Array(d.u.data)
    Tmn, Tmx = Inf32, -Inf32
    nF = nI = nliq = 0
    umax = 0.0f0
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        su == TYPE_I && (nI += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        Tmn = min(Tmn, TA[n])
        Tmx = max(Tmx, TA[n])
        fsA[n] < 0.5f0 && (nliq += 1)
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    depth = liquid_column_depth(model, ix, iy)
    return Tmn, Tmx, nF, nI, nliq, umax, depth
end

export!(model; dir="output_laser_pool")
Tmn0, Tmx0, nF0, nI0, nliq0, umax0, depth0 = pool_probe(model)
@info "laser pool" Nx Ny Nz Hfill n_heat nskin A0 depth0 nliq0 nF0 T_init backend

function run_pool!(model, nsteps, every)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="laser pool ")
    t = 0
    Tmn = Tmx = zero(Float32)
    nF = nI = nliq = depth = 0
    umax = 0.0f0
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_laser_pool")
        Tmn, Tmx, nF, nI, nliq, umax, depth = pool_probe(model)
        next!(prog; showvalues = [
            (:t, t),
            (:Tmax_K, round(si_T(model.units, Tmx); digits=0)),
            (:depth, depth),
            (:nliq, nliq),
            (:nF, nF),
            (:nI, nI),
            (:umax, round(umax; digits=3)),
        ])
    end
    finish!(prog)
    return Tmn, Tmx, nF, nI, nliq, umax, depth
end

Tmn, Tmx, nF, nI, nliq, umax, depth = run_pool!(model, n_heat, every)
@info "laser pool done" Tmax_K=si_T(units, Tmx) depth0 depth nliq0 nliq nF nF0 nI umax
