# DED-style 316L melt track. Square-ish pad with Ny ≈ Nx/2, Δx ≈ 18 µm so a
# 2 kW / 0.5 mm Gaussian is close to D3Q7-stable Q. The beam and powder feed
# walk in x (Q + msrc fields — no laser/powder module). Pad starts at 300 K.
# Evaporation cooling, evaporative mass loss, and recoil are on (latent_v, T_v).
#
#   SURFACE = true, TEMPERATURE = true, VOLUME_FORCE = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_melt_pool")

function set_moving_gaussians!(Qh, Sh, Nx, Ny, Nz, Hfill, zQmin, zSmin,
        x_las, y_las, invw2, invw2S, Q0, S0, rmax)
    fill!(Qh, 0)
    fill!(Sh, 0)
    x0 = max(2, floor(Int, x_las - rmax))
    x1 = min(Nx - 1, ceil(Int, x_las + rmax))
    y0 = max(2, floor(Int, y_las - rmax))
    y1 = min(Ny - 1, ceil(Int, y_las + rmax))
    z1 = min(Hfill + 1, Nz - 1)
    for z in min(zQmin, zSmin):z1, y in y0:y1, x in x0:x1
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        dx = Float32(x) - x_las
        dy = Float32(y) - y_las
        r2 = dx * dx + dy * dy
        z >= zQmin && (Qh[n] = Q0 * exp(-invw2 * r2))
        z >= zSmin && (Sh[n] = S0 * exp(-invw2S * r2))
    end
    return nothing
end

function track_metrics(fsA, flags, TA, uA, Nx, Ny, Nz, Hfill, x_las, y_las, U)
    nliq = 0
    Tmax = -Inf32
    Tmin = Inf32
    umax = 0.0f0
    zI = 0
    nsolid = 4 + (4 - 1) * Nx + (4 - 1) * Nx * Ny
    xl = clamp(round(Int, x_las), 2, Nx - 1)
    yl = clamp(round(Int, y_las), 2, Ny - 1)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        su = flags[n] & TYPE_SU
        su == TYPE_I && (zI = max(zI, z))
        if su == TYPE_F || su == TYPE_I
            umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
            Tmax = max(Tmax, TA[n])
            Tmin = min(Tmin, TA[n])
            fsA[n] < 0.5f0 && (nliq += 1)
        end
    end
    depth = 0
    for z in Hfill:-1:2
        n = xl + (yl - 1) * Nx + (z - 1) * Nx * Ny
        fsA[n] < 0.5f0 || break
        depth += 1
    end
    u_sol = hypot(uA[nsolid, 1], uA[nsolid, 2], uA[nsolid, 3])
    return nliq, depth, Float32(si_T(U, Tmax)), Float32(si_T(U, Tmin)), umax, u_sol, zI, xl
end

# --- SI: 316L DED, fine mesh (Ny ≈ Nx/2) ---
Nx, Ny, Nz = 192, 96, 56
Hfill = 42
L = Hfill - 2                           # pad thickness in cells

si_H      = 0.72e-3u"m"                 # → Δx ≈ 18 µm
si_ρ      = 8000u"kg/m^3"
si_cp     = 500u"J/kg/K"
si_k_s    = 15.0u"W/m/K"
si_k_l    = 30.0u"W/m/K"
si_Tm     = 1673.0u"K"                  # 316L ~ 1400 °C
si_T_init = 300.0u"K"
si_Lheat  = 2.8e5u"J/kg"
si_Lv     = 7.45e6u"J/kg"               # vaporization; 0 → off
si_Tv     = 3086.0u"K"                  # 316L boiling
si_M      = 0.0558u"kg/mol"
si_K0     = 1.0e-10u"m^2"
si_P      = 2000.0u"W"
si_d_spot = 0.5e-3u"m"                  # 1/e² diameter
si_v      = 8.0e-3u"m/s"
si_A      = 0.35
si_δ      = 0.12e-3u"m"                 # physical absorption depth (not 3 cells)
si_mdot   = 2.0u"g/minute"              # 10 g/min needs ~2 mm of gas; this slice has ~0.2 mm
si_eta    = 0.7                         # catchment efficiency
si_σ      = 0.015u"N/m"                 # reduced; 316L ~ 1.8 N/m
si_σT     = -1.5e-5u"N/m/K"
q_max     = 0.04f0

α_l_si = si_k_l / (si_ρ * si_cp)
α_s_si = si_k_s / (si_ρ * si_cp)
si_ν_l = ustrip(u"m^2/s", α_l_si) * u"m^2/s"
si_ν_s = 0.5 * si_ν_l
lbm_α_true = 0.1
m = ustrip(u"m", si_H) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_l_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"

units = Units(si_H, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

w_m    = 0.5 * ustrip(u"m", si_d_spot)
I_peak = 2 * ustrip(u"W", si_P) / (π * w_m^2)
δ_m    = ustrip(u"m", si_δ)
Q_si   = si_A * I_peak / δ_m
Q_full = Float32(lbm_Q(units, Q_si * u"W/m^3"))
scale  = Q_full > q_max ? Float32(q_max / Q_full) : 1.0f0
Q0     = scale * Q_full
P_used = scale * si_A * ustrip(u"W", si_P)
mdot_kg_s = si_eta * ustrip(u"kg/s", uconvert(u"kg/s", si_mdot))
# 1/e² Gaussian ∫exp(-2 r²/w²) dA = π w²/2; spread over absorption depth δ.
S_peak = mdot_kg_s / (0.5 * π * w_m^2 * δ_m)   # kg/m³/s at the axis
S0     = Float32(lbm_S(units, S_peak))

model = Model(Nx, Ny, Nz, units;
              ν = si_ν_l,
              α = 2 * α_l_si,
              α_s = 2 * α_s_si,
              α_l = 2 * α_l_si,
              ν_s = si_ν_s,
              ν_l = si_ν_l,
              β = 0.0f0, gz = 0.0f0,
              σ = si_σ, σT = si_σT, Tσ = si_Tm,
              latent = si_Lheat,
              Ts = si_Tm, Tl = si_Tm, K0 = si_K0,
              latent_v = si_Lv, T_v = si_Tv, M = si_M,
              T_avg = Float32(lbm_T(units, si_Tm)),
              backend = CUDABackend())

Tm      = Float32(lbm_T(units, si_Tm))
T_init  = Float32(lbm_T(units, si_T_init))
w_cells = w_m / Float64(units.m)
invw2   = w_cells > 0 ? 2.0f0 / Float32(w_cells^2) : 0.0f0
invw2S  = invw2 / 1.44f0                    # powder slightly wider than the beam
v_lat   = Float32(ustrip(u"m/s", si_v) * units.s / units.m)
y_las   = Float32(Ny + 1) / 2
x0      = Float32(8 + 2 * w_cells)
x1      = Float32(Nx - 7 - 2 * w_cells)
nsteps  = max(2, round(Int, (x1 - x0) / max(v_lat, Float32(1e-8))))
qevery  = max(1, round(Int, 0.25f0 / max(v_lat, Float32(1e-8))))  # ~1/4 cell
every   = max(qevery, nsteps ÷ 40)
nδ      = max(2, round(Int, δ_m / m))
zQmin   = max(2, Hfill - nδ + 1)
zSmin   = zQmin
rmax    = Float32(4 * w_cells * 1.2)
σlat    = model.domains[1].σ
if scale < 1
    @warn "2 kW / 0.5 mm maps to Q_lat=$(Q_full); capped to $(Q0) (absorbed ≈ $(round(P_used; digits=1)) W)" Q_full Q0 scale P_used
else
    @info "using full 2 kW (Q_lat=$(Q_full) ≤ q_max=$(q_max))" Q_full
end
if σlat > 0.05f0
    @warn "lattice σ=$(σlat) is large; CSF may blow up. Lower si_σ." σlat
end
@info "DED 316L fine track" Nx Ny Nz m_um=(1e6*m) w_cells v_lat nsteps qevery Ncell=(Nx*Ny*Nz) si_P si_v si_mdot si_eta S0 Λ_v=model.domains[1].Λ_v T_v=model.domains[1].T_v T_init

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_init, Nx * Ny * Nz)
fsh  = ones(Float32, Nx * Ny * Nz)
Qh   = zeros(Float32, Nx * Ny * Nz)
Sh   = zeros(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
        host[n] = TYPE_S
    elseif z <= Hfill
        host[n] = TYPE_F
        Th[n] = T_init
        fsh[n] = 1
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
set_moving_gaussians!(Qh, Sh, Nx, Ny, Nz, Hfill, zQmin, zSmin,
                      x0, y_las, invw2, invw2S, Q0, S0, rmax)
copyto!(model.domains[1].Q.data, Qh)
copyto!(model.domains[1].msrc.data, Sh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_melt_pool")

nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.3, desc="DED fine ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        ninner = every
        nq = qevery
        for j in 1:ninner
            if (j - 1) % nq == 0
                x_las = x0 + v_lat * Float32(Int(d.t))
                set_moving_gaussians!(Qh, Sh, Nx, Ny, Nz, Hfill, zQmin, zSmin,
                                      x_las, y_las, invw2, invw2S, Q0, S0, rmax)
                copyto!(d.Q.data, Qh)
                copyto!(d.msrc.data, Sh)
            end
            run!(model, 1)
        end
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_melt_pool")
    t_lat = Int(d.t)
    x_now = x0 + v_lat * Float32(t_lat)
    nliq, depth, Tmax_K, Tmin_K, umax, _, zI, xl = track_metrics(
        Array(d.fs.data), Array(d.flags.data), Array(d.T.data), Array(d.u.data),
        Nx, Ny, Nz, Hfill, x_now, y_las, model.units)
    next!(prog; showvalues = [
        (:t, t_lat),
        (:t_si, round(si_t(model.units, t_lat); digits=4)),
        (:x_las, round(xl; digits=1)),
        (:nliq, nliq),
        (:depth, depth),
        (:Tmax_K, round(Tmax_K; digits=1)),
        (:Tmin_K, round(Tmin_K; digits=1)),
        (:umax, round(umax; digits=3)),
        (:zI, zI),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)

LatticeBoltzmann.moments!(model)
x_end = x0 + v_lat * Float32(nsteps)
nliq, depth, Tmax_K, Tmin_K, umax, u_sol, zI, xl = track_metrics(
    Array(d.fs.data), Array(d.flags.data), Array(d.T.data), Array(d.u.data),
    Nx, Ny, Nz, Hfill, x_end, y_las, model.units)
@info "DED fine report" nliq depth_cells=depth Tmax_K Tmin_K umax u_sol zI Hfill x_las=xl t_end=si_t(model.units, nsteps) P_used_W=P_used scale Q_full S0 mdot_kg_s
