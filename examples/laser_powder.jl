# Step 6 of laser–foam coupling: mixed 316L-like + foaming-agent powder.
# Pad starts with a=0. Laser opens a pool, then a Gaussian jet deposits metal
# plus agent_frac into a. Arrhenius dumps a→c only in the liquid. No scan.
#
# Tests: test/test_laser_powder.jl, test/test_blowing_agent.jl (agent_frac)
# ParaView: output_laser_powder/lbm.pvd. Colour **a** (from the jet) and **c**.
# fs is the pool. Threshold flags 8–32, Contour phi=0.5.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_laser_powder")

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
si_T_init = 0.88 * si_Tm
si_Lheat = 2.7e5u"J/kg"
si_Lv = 7.45e6u"J/kg"
si_Tv = 3086.0u"K"
si_M = 0.0558u"kg/mol"
si_σ = 1.5u"N/m"
si_P = 280.0u"W"
si_d_spot = 0.5e-3u"m"
si_δ = 0.60e-3u"m"
si_t_heat = 4.0e-3u"s"
si_t_powder_delay = 1.5e-3u"s"
si_k_a = 2.0e5u"s^-1"
si_E_a = 6.7e3u"K"
si_mdot = 1.0u"g/minute"
si_v_jet = 8.0u"m/s"
si_agent_frac = 0.04
k_H = 3.0f0

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
n_delay = max(0, round(Int, ustrip(u"s", si_t_powder_delay) / s))
every = max(10, n_heat ÷ 8)

T_init = Float32(lbm_T(units, si_T_init))
nskin = max(3, round(Int, ustrip(u"m", si_δ) / m))
backend = CUDA.functional() ? CUDABackend() : CPU()

model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_l_si, α_s=2 * α_s_si, α_l=2 * α_l_si, α_c=α_l_si,
              ν_s=0.5 * si_ν, ν_l=si_ν, k_H=k_H,
              gz=0, σ=si_σ,
              k_a=si_k_a, E_a=si_E_a, Y_a=1, a_fs_max=0.5,
              T_avg=T_init, latent=si_Lheat, Ts=si_Tm, Tl=si_Tm,
              latent_v=si_Lv, T_v=si_Tv, M=si_M,
              emissivity=0.4, T_rad=si_T_init,
              powder_τ=0, powder_T=si_T_init,
              K0=1.0e-10u"m^2",
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64)

xc = (Nx + 1) / 2
yc = (Ny + 1) / 2
model.laser = Laser(units; P=si_P, w=0.5 * si_d_spot,
                    x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                    nrays=11, max_bounce=4, every=1, skin=nskin)
model.powder_jet = PowderJet(units; mdot=si_mdot, w=0.6 * si_d_spot, v=si_v_jet,
                             x=xc, y=yc, z=Float32(Nz) - 1.2f0,
                             dir=(0, 0, -1), nparcels=12,
                             enabled=false, agent_frac=si_agent_frac)

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

function powder_probe(model)
    d = model.domains[1]
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    aA = Array(d.a.data)
    cA = Array(d.c.data)
    uA = Array(d.u.data)
    Tmx = -Inf32
    nliq = 0
    a_liq = a_tot = 0.0
    c_liq_max = 0.0f0
    umax = 0.0f0
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        Tmx = max(Tmx, TA[n])
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        a_tot += Float64(aA[n])
        if fsA[n] < 0.5f0
            nliq += 1
            a_liq += Float64(aA[n])
            c_liq_max = max(c_liq_max, cA[n])
        end
    end
    depth = liquid_column_depth(model, ix, iy)
    inv = agent_inventory(d)
    nfar = 4 + 3 * Int(d.Nx) + 3 * Int(d.Nx) * Int(d.Ny)
    return (Tmx, nliq, depth, umax, a_tot,
            nliq == 0 ? 0.0 : a_liq / nliq,
            aA[nfar], c_liq_max, inv.res, inv.a)
end

export!(model; dir="output_laser_powder")
p0 = powder_probe(model)
@info "laser powder" Nx Ny Nz Hfill n_heat n_delay nskin si_agent_frac depth0=p0[3] backend

function run_powder!(model, nsteps, every, n_delay)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="laser powder ")
    t = 0
    Tmx = zero(Float32)
    nliq = depth = 0
    umax = 0.0f0
    a_tot = a_liq = a_far = 0.0
    c_liq_max = zero(Float32)
    a_res = a_inv = zero(Float32)
    while t < nsteps
        if t >= n_delay && model.powder_jet !== nothing
            model.powder_jet.enabled = true
        end
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_laser_powder")
        Tmx, nliq, depth, umax, a_tot, a_liq, a_far, c_liq_max, a_res, a_inv =
            powder_probe(model)
        next!(prog; showvalues = [
            (:t, t),
            (:jet, model.powder_jet.enabled),
            (:Tmax_K, round(si_T(model.units, Tmx); digits=0)),
            (:depth, depth),
            (:nliq, nliq),
            (:a_tot, round(a_tot; digits=3)),
            (:a_liq, round(a_liq; digits=4)),
            (:a_far, round(a_far; digits=5)),
            (:c_liq, round(c_liq_max; digits=3)),
        ])
    end
    finish!(prog)
    return Tmx, nliq, depth, umax, a_tot, a_liq, a_far, c_liq_max, a_res, a_inv
end

Tmx, nliq, depth, umax, a_tot, a_liq, a_far, c_liq_max, a_res, a_inv =
    run_powder!(model, n_heat, every, n_delay)
@info "laser powder done" Tmax_K=si_T(units, Tmx) depth nliq a_tot a_liq a_far c_liq_max a_res umax
