using Test
using LatticeBoltzmann
using KernelAbstractions

# 32×4×32 is 3D. DIM=2 rejects Nz != 1, so this testset is not compiled there.
# The counted event is a later crossing of the start height, or a landing on it.
@static if DIM == 3
function wave_half_height(phi, Nx, Ny, Nz, x::Int)
    # Two interfaces exist. The crest is the φ = 0.5 crossing on the spec sine,
    # not the first crossing a bottom-up scan meets.
    z_ref = Nz / 2 + sin(2π * x / Nx)
    acc = 0.0
    for y in 0:Ny-1
        best = NaN
        best_d = Inf
        for z in 0:Nz-1
            z1 = z == Nz - 1 ? 0 : z + 1
            p0 = Float64(phi[x + y * Nx + z * Nx * Ny + 1])
            p1 = Float64(phi[x + y * Nx + z1 * Nx * Ny + 1])
            if (p0 - 0.5) * (p1 - 0.5) <= 0 && p1 != p0
                h = z + (0.5 - p0) / (p1 - p0)
                if z == Nz - 1 && h >= Nz - 0.5
                    h -= Nz
                end
                d = abs(h - z_ref)
                if d < best_d
                    best_d = d
                    best = h
                end
            end
        end
        # A column with no φ = 0.5 crossing is a miss, not the box mid-plane.
        if !isfinite(best)
            return NaN
        end
        acc += best
    end
    return acc / Ny
end

# h0 is the initial φ = 0.5 height. Crossing z = Nz/2 only means the crest has
# left the start side. The event is the next strict crossing of h0, or the
# first sample that lands on h0. Neither flat-level crossing is that event.
function wave_event_step(heights, h0, flat)
    start_above = h0 >= flat
    left = false
    prev_h = h0
    for (i, h) in enumerate(heights)
        if !isfinite(h)
            return nothing
        end
        on_other = start_above ? h < flat : h > flat
        if on_other
            left = true
        end
        if left && (h == h0 || (prev_h - h0) * (h - h0) < 0)
            return i
        end
        prev_h = h
    end
    return nothing
end

@testset "Allen–Cahn capillary wave" begin
    backend = CPU()
    σ = 0.01
    rho_a = 1.0
    rho_b = 2.0
    Nx, Ny, Nz = 32, 4, 32
    model = Model(Nx, Ny, Nz, 0.01; backend=backend, workgroup=64,
                  σ=σ, W=4, Mphi=0.05, rho_a=rho_a, rho_b=rho_b)
    domain = model.domains[1]
    @test domain.β == 0
    @test domain.fx == 0 && domain.fy == 0 && domain.fz == 0
    W = Float64(domain.W)
    phi = domain.phi.data
    # grad_phi is periodic and does not read flags. A raw tanh(2z/W) is a
    # one-cell jump at the seam. Wrapping each signed distance into (-Nz/2, Nz/2]
    # before the tanh makes z = 0 one equilibrium interface. The crest stays on
    # the spec sine, so the heavy fluid under it is half the box.
    half = Float64(Nz) / 2
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        z_crest = Nz / 2 + sin(2π * x / Nx)
        s0 = mod(z + half, Nz) - half
        s0 = s0 <= -half ? half : s0
        sc = mod((z_crest - z) + half, Nz) - half
        sc = sc <= -half ? half : sc
        s = abs(s0) <= abs(sc) ? s0 : sc
        n = x + y * Nx + z * Nx * Ny + 1
        phi[n] = (1 + tanh(2 * s / W)) / 2
    end
    initialize!(model)

    k = 2π / Nx
    ω = sqrt(σ * k^3 / (rho_a + rho_b))
    Tper = 2π / ω
    h0 = wave_half_height(phi, Nx, Ny, Nz, 1)
    z_spec = Nz / 2 + sin(2π * 1 / Nx)
    @test isfinite(h0)
    @test abs(h0 - z_spec) < 0.5
    flat = Nz / 2
    heights = Vector{Float64}(undef, 1500)
    for step in 1:1500
        LatticeBoltzmann.step!(model)
        heights[step] = wave_half_height(domain.phi.data, Nx, Ny, Nz, 1)
    end
    event = wave_event_step(heights, h0, flat)
    println("wave_event_step=", event, " T=", Tper,
            " ratio=", event === nothing ? nothing : event / Tper,
            " h0=", h0, " h_event=", event === nothing ? nothing : heights[event])
    @test all(isfinite, heights)
    @test event !== nothing && 0.7 * Tper <= event <= 1.5 * Tper
    @test all(isfinite, domain.fi.data)
    @test all(isfinite, domain.hi.data)
end
end # DIM == 3
