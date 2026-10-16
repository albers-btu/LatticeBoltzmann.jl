# 2D drop falling onto an open liquid surface.
# φ = 1 is the liquid (drop and pool). φ = 0 is the gas. The free surface is
# the φ = 0.5 contour. Both fluids are solved: Saclay NS with F = μ∇φ + ρ(φ) g,
# and conservative Allen–Cahn (λ = 0) with the equilibrium counter-term.
#
# Height is y, so gravity is fy. Bounce-back walls close the box. The open
# surface is the pool's φ = 0.5 contour, not a contact-angle condition.
#
# Density ratio ρ_liquid/ρ_gas = 10. The gas kinematic viscosity is higher
# than the liquid's so the film under the drop stays well below c_s on TRT.
# Saclay's ratio-100 capillary wave uses central moments.
#
# Set Preferences, then start a new Julia process:
#   interface => "allen_cahn"
#   dim => 2
# This script does not set those preferences.
#
#   output_allen_cahn_drop/lbm.pvd
# The grid is one cell thick, so ParaView can draw it. Open that file,
# color by phi, and add a Contour at 0.5. Walls are flags = 1.

using LatticeBoltzmann
using KernelAbstractions
using Printf

@assert ALLEN_CAHN
@assert DIM == 2

# Liquid above the pool. y_cut sits in the gas gap, above the pool tails.
function liquid_above(phi, u, Nx, Ny, y_cut)
    m = 0.0
    sx = 0.0
    sy = 0.0
    su = 0.0
    for y in (y_cut + 1):(Ny - 1), x in 1:(Nx - 2)
        n = x + y * Nx + 1
        φ = Float64(phi[n])
        φ <= 0.5 && continue
        m += φ
        sx += φ * x
        sy += φ * y
        su += φ * Float64(u[n, 2])
    end
    m == 0 && return (NaN, NaN, 0.0, NaN)
    return (sx / m, sy / m, m, su / m)
end

# Underside of the drop on one column: the φ = 0.5 crossing above the pool.
# NaN once that crossing has joined the pool.
function drop_bottom(phi, Nx, Ny, x)
    seen_pool = false
    for y in 1:(Ny - 2)
        n0 = x + (y - 1) * Nx + 1
        n1 = x + y * Nx + 1
        p0 = Float64(phi[n0])
        p1 = Float64(phi[n1])
        if !seen_pool && p0 >= 0.5 && p1 < 0.5
            seen_pool = true
            continue
        end
        if seen_pool && p0 < 0.5 && p1 >= 0.5
            return (y - 1) + (0.5 - p0) / (p1 - p0)
        end
    end
    return NaN
end

function max_speed(u)
    umax = 0.0
    @inbounds for n in 1:size(u, 1)
        s = sqrt(Float64(u[n, 1])^2 + Float64(u[n, 2])^2 + Float64(u[n, 3])^2)
        umax = s > umax ? s : umax
    end
    return umax
end

function fields_finite(phi, u)
    @inbounds for n in eachindex(phi)
        isfinite(phi[n]) || return false
    end
    @inbounds for n in 1:size(u, 1)
        (isfinite(u[n, 1]) && isfinite(u[n, 2]) && isfinite(u[n, 3])) || return false
    end
    return true
end

let
    Nx, Ny, Nz = 64, 96, 1
    ν = 0.02
    ν_gas = 0.10
    σ = 1e-3
    g = 4e-5
    R = 8.0
    Wφ = 4.0
    ys = 28.0                 # pool surface; y increases upward
    cx = (Nx - 1) / 2
    cy = 62.0                 # drop center, one diameter of gas above the pool
    y_cut = 38                # gas gap used to track the drop
    xcol = round(Int, cx)

    model = Model(Nx, Ny, Nz, ν; backend=CPU(), workgroup=64,
                  fy=-g, σ=σ, β=0, σT=0, K0=0,
                  W=Wφ, Mphi=0.02, rho_a=0.1, rho_b=1,
                  nu_a=ν_gas, nu_b=ν)
    domain = model.domains[1]
    phi = domain.phi.data
    flags = domain.flags.data
    for y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = x + y * Nx + 1
        if x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1
            flags[n] = TYPE_S
        end
        r = sqrt((x - cx)^2 + (y - cy)^2)
        φ_pool = (1 + tanh(2 * (ys - y) / Wφ)) / 2
        φ_drop = (1 + tanh(2 * (R - r) / Wφ)) / 2
        phi[n] = max(φ_pool, φ_drop)
    end
    initialize!(model)

    x0, y0, m0, _ = liquid_above(phi, domain.u.data, Nx, Ny, y_cut)
    b0 = drop_bottom(phi, Nx, Ny, xcol)
    m0 > 10 || error("drop is not above the pool at t = 0 (mass $m0)")
    50 < b0 < 58 || error("drop underside starts at y = $b0, expected near 54")

    dir = "output_allen_cahn_drop"
    fields = (:phi, :u, :flags)
    export!(model; dir=dir, fields=fields, sync=true)

    nsteps = 8000
    every = 400
    b_end = b0
    @printf("t = %4d  drop (x, y) = (%6.2f, %6.2f)  underside = %6.2f  mass = %7.1f  |u|max = %.3e\n",
            0, x0, y0, b0, m0, max_speed(domain.u.data))
    for chunk in 1:(nsteps ÷ every)
        for _ in 1:every
            LatticeBoltzmann.step!(model)
        end
        xc, yc, mc, _ = liquid_above(phi, domain.u.data, Nx, Ny, y_cut)
        b_end = drop_bottom(phi, Nx, Ny, xcol)
        umax = max_speed(domain.u.data)
        t = chunk * every
        @printf("t = %4d  drop (x, y) = (%6.2f, %6.2f)  underside = %6.2f  mass = %7.1f  |u|max = %.3e\n",
                t, xc, yc, b_end, mc, umax)
        export!(model; dir=dir, fields=fields, sync=true)
        fields_finite(phi, domain.u.data) ||
            error("non-finite φ or u at t = $t")
        umax < 0.25 ||
            error("speed |u| = $umax is too close to c_s at t = $t")
    end

    landed = !isfinite(b_end) || b_end < ys + Wφ
    y_final = isfinite(b_end) ? b_end : ys
    fall = b0 - y_final
    @printf("underside fell %.2f cells (y %.2f → %s)\n",
            fall, b0, isfinite(b_end) ? @sprintf("%.2f", b_end) : "pool")
    landed ||
        error("drop underside is still at y = $(round(b_end; digits=2)) after $nsteps steps")
    fall >= 15 ||
        error("drop underside fell only $(round(fall; digits=2)) cells")
    println("Wrote $dir/lbm.pvd (t = 0, $every, …, $nsteps)")
    println("ParaView: color by phi and contour at 0.5.")
end
