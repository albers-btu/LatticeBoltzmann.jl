# Short one-layer DED track: moving laser, mixed powder, Arrhenius, then nuclei
# in the wake. Not the full 8 mm/s / 2 kW melt_pool scan. Pad ~3.2 × 2.4 × 1.2 mm,
# 1.6 mm pass at 80 mm/s, 400 W. Numerical skin 12 → 4 after a pool exists (keep
# P). Nuclei after the pass with the beam off — v_lat is tiny (capillary Δt), so
# a 3³ hole under a live spot is a crater.
#
# Tests: test/test_ded_track.jl
# ParaView: output_ded_track/lbm.pvd. Colour fs (trail), a (jet), c (dissolved).
# Contour phi=0.5 (nuclei). Threshold flags 8–32.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_ded_track")

si_Lx = 3.2e-3u"m"
si_Ly = 2.4e-3u"m"
si_H  = 1.2e-3u"m"
si_gas = 0.8e-3u"m"
si_dx = 50.0e-6u"m"
si_end_margin = 0.8e-3u"m"

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
si_P = 400.0u"W"
si_d_spot = 0.5e-3u"m"
si_δ_melt = 0.60e-3u"m"
si_δ_nuc = 0.20e-3u"m"
si_v = 80.0e-3u"m/s"
si_t_powder_delay = 1.5e-3u"s"
si_t_melt = 4.0e-3u"s"              # skin 12 → 4 after a pool can exist
si_k_a = 2.0e5u"s^-1"
si_E_a = 6.7e3u"K"
si_mdot = 0.1u"g/s"
si_v_jet = 8.0u"m/s"
si_agent_frac = 0.15
k_H = 3.0f0
jet_along = :back

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
T_init = Float32(lbm_T(units, si_T_init))
nskin_melt = max(3, round(Int, ustrip(u"m", si_δ_melt) / m))
nskin_nuc = max(3, round(Int, ustrip(u"m", si_δ_nuc) / m))
w_m = 0.5 * ustrip(u"m", si_d_spot)
w_cells = w_m / m
v_lat = Float32(ustrip(u"m/s", si_v) * s / m)
y_las = Float32(Ny + 1) / 2
n_end = ustrip(u"m", si_end_margin) / m
x0 = Float32(clamp(n_end, 4.0, Nx / 4))
x1 = Float32(clamp(Nx + 1 - n_end, 3 * Nx / 4, Nx - 3))
nsteps_pass = max(2, round(Int, abs(x1 - x0) / max(v_lat, Float32(1e-8))))
n_delay = si_t_powder_delay > 0u"s" ?
    max(0, round(Int, ustrip(u"s", si_t_powder_delay) / s)) : 0
n_melt = max(n_delay, round(Int, ustrip(u"s", si_t_melt) / s))
n_nuc = 16
every = max(8, nsteps_pass ÷ 16)
dx_noz = Float32(max(0.5e-3 / m, 2 * w_cells))
z_noz = Float32(Nz) - 1.4f0
z_aim = Float32(Hfill)
backend = CUDA.functional() ? CUDABackend() : CPU()

nuc = Nucleation{Float32}(; enabled=false, d_min=8, R=1, c_star=1.08f0,
                          p_cell=1, n_max=3, n_over=1.0f0, every=4,
                          n_total_max=8)
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
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))

model.laser = Laser(units; P=si_P, w=0.5 * si_d_spot,
                    x=x0, y=y_las, z=Float32(Nz) - 1.1f0,
                    nrays=11, max_bounce=4, every=1, skin=nskin_melt)
model.powder_jet = PowderJet(units; mdot=si_mdot, w=0.6 * si_d_spot, v=si_v_jet,
                             x=x0, y=y_las, z=z_noz,
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

function place_powder_jet!(jet, x_las, y_las, z_noz, z_aim, sgn, dx_noz, jet_along, Nx)
    along = jet_along === :front ? sgn : -sgn
    x_noz = clamp(x_las + along * dx_noz, 2.5f0, Float32(Nx) - 1.5f0)
    set_powder_jet_position!(jet, x_noz, y_las, z_noz)
    aim_powder_jet!(jet, x_las, y_las, z_aim)
    return jet
end

function mix_liquid_c!(model, cmin=1.20f0)
    d = model.domains[1]
    cA = Array(d.c.data)
    fsA = Array(d.fs.data)
    fl = Array(d.flags.data)
    @inbounds for n in eachindex(cA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        fsA[n] < 0.99f0 || continue
        cA[n] = max(cA[n], cmin)
    end
    copyto!(d.c.data, cA)
    return nothing
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

function track_probe(model, x_las, y_las)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    aA = Array(d.a.data)
    cA = Array(d.c.data)
    uA = Array(d.u.data)
    Tmx = -Inf32
    nliq = nF = n_trail = 0
    a_tot = 0.0
    c_liq_max = 0.0f0
    umax = 0.0f0
    xmin = Nx
    xmax = 1
    yl = clamp(round(Int, y_las), 2, Ny - 1)
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        Tmx = max(Tmx, TA[n])
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        a_tot += Float64(aA[n])
        if fsA[n] < 0.5f0
            nliq += 1
            c_liq_max = max(c_liq_max, cA[n])
        end
    end
    @inbounds for x in 2:(Nx - 1)
        has = false
        for z in 2:(Nz - 1)
            n = x + (yl - 1) * Nx + (z - 1) * Nx * Ny
            su = fl[n] & TYPE_SU
            (su == TYPE_F || su == TYPE_I) || continue
            fsA[n] < 0.5f0 || continue
            has = true
            break
        end
        if has
            n_trail += 1
            xmin = min(xmin, x)
            xmax = max(xmax, x)
        end
    end
    xl = clamp(round(Int, x_las), 2, Nx - 1)
    depth = liquid_column_depth(fl, fsA, Nx, Ny, Nz, xl, yl)
    fm = foam_metrics(model)
    nfar = 4 + 3 * Nx + 3 * Nx * Ny
    return (Tmx, nliq, depth, umax, a_tot, aA[nfar], c_liq_max,
            fm.nb, fm.n_planted, fm.a_res, bubble_z_mean(model), nF,
            n_trail, xmin, xmax)
end

place_powder_jet!(model.powder_jet, x0, y_las, z_noz, z_aim, 1.0f0, dx_noz, jet_along, Nx)
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_ded_track")
P0 = model.laser.P
@info "DED short track" Nx Ny Nz Hfill x0 x1 nsteps_pass n_delay n_melt n_nuc nskin_melt nskin_nuc si_P si_v si_agent_frac scan_mm=(1e3 * m * abs(x1 - x0)) backend

function run_track!(model, x0, x1, y_las, v_lat, nsteps_pass, n_delay, n_melt,
                    every, nskin_nuc, Nx, z_noz, z_aim, dx_noz, jet_along, P0)
    xmin, xmax = min(x0, x1), max(x0, x1)
    sgn = one(Float32)
    Tmx = zero(Float32)
    nliq = depth = nb = np = nF = n_trail = xmin_t = xmax_t = 0
    umax = a_tot = a_far = zbar = 0.0
    c_liq_max = a_res = zero(Float32)
    x_now = x0
    nF0 = 0
    prog = Progress(cld(nsteps_pass, every); dt=0.3, desc="DED track ")
    for k in 0:(nsteps_pass - 1)
        x_now = clamp(x0 + sgn * v_lat * Float32(k), xmin, xmax)
        set_laser_position!(model.laser, x_now, y_las)
        jet = model.powder_jet
        if k >= n_delay
            jet.enabled = true
            place_powder_jet!(jet, x_now, y_las, z_noz, z_aim, sgn, dx_noz, jet_along, Nx)
        else
            jet.enabled = false
        end
        if k == n_melt
            model.laser.skin = nskin_nuc
        end
        @assert model.laser.P == P0
        with_logger(NullLogger()) do
            run!(model, 1)
        end
        if (k + 1) % every == 0 || k + 1 == nsteps_pass
            export!(model; dir="output_ded_track")
            Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res, zbar, nF,
                n_trail, xmin_t, xmax_t = track_probe(model, x_now, y_las)
            nF0 == 0 && (nF0 = nF)
            next!(prog; showvalues = [
                (:k, k + 1),
                (:x_las, round(x_now; digits=1)),
                (:P, model.laser.P),
                (:skin, model.laser.skin),
                (:Tmax_K, round(si_T(model.units, Tmx); digits=0)),
                (:depth, depth),
                (:nliq, nliq),
                (:trail, n_trail),
                (:a_tot, round(a_tot; digits=2)),
                (:c_liq, round(c_liq_max; digits=3)),
                (:nb, nb),
                (:umax, round(umax; digits=3)),
            ])
        end
    end
    finish!(prog)
    return (x_now, Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res,
            zbar, nF, nF0, n_trail, xmin_t, xmax_t)
end

function nucleate_wake!(model, n_nuc, every, x_las, y_las)
    model.laser.enabled = false
    fill!(model.domains[1].Q.data, 0)
    jet = model.powder_jet
    jet !== nothing && (jet.enabled = false)
    mix_liquid_c!(model)
    model.bubbles.nucleation.enabled = true
    Tmx = zero(Float32)
    nliq = depth = nb = np = nF = n_trail = xmin_t = xmax_t = 0
    umax = a_tot = a_far = zbar = 0.0
    c_liq_max = a_res = zero(Float32)
    prog = Progress(cld(n_nuc, every); dt=0.3, desc="nucleate ")
    t = 0
    while t < n_nuc
        nrun = min(every, n_nuc - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_ded_track")
        Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res, zbar, nF,
            n_trail, xmin_t, xmax_t = track_probe(model, x_las, y_las)
        next!(prog; showvalues = [
            (:t, t),
            (:nliq, nliq),
            (:trail, n_trail),
            (:nb, nb),
            (:planted, np),
            (:umax, round(umax; digits=3)),
        ])
    end
    finish!(prog)
    return (Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res,
            zbar, nF, n_trail, xmin_t, xmax_t)
end

s1 = run_track!(model, x0, x1, y_las, v_lat, nsteps_pass, n_delay, n_melt,
                every, nskin_nuc, Nx, z_noz, z_aim, dx_noz, jet_along, P0)
x_end, Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res, zbar, nF, nF0,
    n_trail, xmin_t, xmax_t = s1
@info "DED pass done" x_end Tmax_K=si_T(units, Tmx) depth nliq trail=n_trail trail_x=(xmin_t, xmax_t) a_tot c_liq=c_liq_max umax P=model.laser.P nF nF0
s2 = nucleate_wake!(model, n_nuc, min(8, n_nuc), x_end, y_las)
Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res, zbar, nF,
    n_trail, xmin_t, xmax_t = s2
@info "DED short track done" x_end Tmax_K=si_T(units, Tmx) depth nliq trail=n_trail trail_x=(xmin_t, xmax_t) a_tot a_far c_liq=c_liq_max nb planted=np z_bub=zbar umax P=P0 skin=model.laser.skin nF nF0 laser=model.laser.enabled

