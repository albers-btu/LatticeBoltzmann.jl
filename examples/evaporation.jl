# Stationary hot spot with Hertz–Knudsen surface evaporation on TYPE_I.
# Without L_v, T runs away. With L_v, Tmax levels off near T_v.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_evaporation")

function tmax_FI(TA, flags)
    Tmax = -Inf32
    for n in eachindex(TA)
        su = flags[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) && (Tmax = max(Tmax, TA[n]))
    end
    return Tmax
end

Nx, Ny, Nz = 24, 12, 24
Hfill = 14
L = Hfill - 2

si_H      = 0.003u"m"
si_ρ      = 8000u"kg/m^3"
si_cp     = 500u"J/kg/K"
si_k      = 30.0u"W/m/K"
si_Tm     = 1673.0u"K"
si_T_init = si_Tm
si_Tv     = 3086.0u"K"                  # 316L boiling
si_Lheat  = 2.8e5u"J/kg"
si_Lv     = 7.45e6u"J/kg"
si_M      = 0.0558u"kg/mol"
si_Q_peak = 8.0e10u"W/m^3"
si_w      = 0.6e-3u"m"
si_t_end  = 0.4u"s"

α_si = si_k / (si_ρ * si_cp)
si_ν = ustrip(u"m^2/s", α_si) * u"m^2/s"
lbm_α_true = 0.1
m = ustrip(u"m", si_H) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_si)
si_u = (0.05 * m / s) * u"m/s"
units = Units(si_H, si_u, si_ρ; x=L, u=0.05, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

model = Model(Nx, Ny, Nz, units;
              ν = si_ν, α = 2 * α_si, α_s = 2 * α_si, α_l = 2 * α_si,
              β = 0, gz = 0, σ = 0,
              latent = si_Lheat, Ts = si_Tm, Tl = si_Tm, K0 = 1.0e-10u"m^2",
              latent_v = si_Lv, T_v = si_Tv, M = si_M,
              T_avg = Float32(lbm_T(units, si_Tm)),
              backend = CUDABackend())

Tm = Float32(lbm_T(units, si_Tm))
Tv = Float32(lbm_T(units, si_Tv))
Q0 = Float32(lbm_Q(units, si_Q_peak))
w_cells = ustrip(u"m", si_w) / Float64(units.m)
nsteps = max(2, round(Int, ustrip(u"s", si_t_end) / Float64(units.s)))
every = max(50, nsteps ÷ 20)
xc = Float32(Nx + 1) / 2
yc = Float32(Ny + 1) / 2
@info "evaporation spot" Tm Tv Q0 Λ_v=model.domains[1].Λ_v nsteps

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(Tm, Nx * Ny * Nz)
fsh = zeros(Float32, Nx * Ny * Nz)
Qh = zeros(Float32, Nx * Ny * Nz)
invw2 = w_cells > 0 ? 2.0f0 / Float32(w_cells^2) : 0.0f0
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
        host[n] = TYPE_S
    elseif z <= Hfill
        host[n] = TYPE_F
        dx = Float32(x) - xc
        dy = Float32(y) - yc
        Qh[n] = Q0 * exp(-invw2 * (dx * dx + dy * dy))
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
copyto!(model.domains[1].Q.data, Qh)
d = model.domains[1]
LatticeBoltzmann.initialize!(model)
fl = Array(d.flags.data)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if (fl[n] & TYPE_SU) == TYPE_I
        dx = Float32(x) - xc
        dy = Float32(y) - yc
        Qh[n] = Q0 * exp(-invw2 * (dx * dx + dy * dy))
    end
end
copyto!(d.Q.data, Qh)
export!(model; dir="output_evaporation")

nchunks = nsteps ÷ every
prog = Progress(nchunks; dt=0.2, desc="evaporation ", showspeed=true)
for i in 1:nchunks
    with_logger(NullLogger()) do
        run!(model, every)
    end
    LatticeBoltzmann.moments!(model)
    Tmax = tmax_FI(Array(d.T.data), Array(d.flags.data))
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:Tmax_K, round(si_T(model.units, Tmax); digits=1)),
        (:Tv_K, round(si_T(model.units, Tv); digits=1)),
    ])
    export!(model; dir="output_evaporation")
end
finish!(prog)
Tmax = tmax_FI(Array(d.T.data), Array(d.flags.data))
@info "evaporation report" Tmax_K=si_T(model.units, Tmax) Tv_K=si_T(model.units, Tv)
