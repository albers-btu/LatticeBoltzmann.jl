using Test, LatticeBoltzmann
using KernelAbstractions: CPU

@testset "foam allocation and no-op step" begin
    @test FOAM

    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    N = length(domain.flags)
    @test hasproperty(domain, :ci)
    @test hasproperty(domain, :tag)
    @test length(domain.ci) == length(domain.gi) == 7 * N
    @test length(domain.c) == N
    @test length(domain.ϕ_old) == N
    @test length(domain.ρb) == N
    @test length(domain.Pi) == N
    @test length(domain.tag) == N
    @test eltype(domain.tag) == Int32
    @test length(domain.flux) == 4096
    @test all(iszero, Array(domain.ϕ_old.data))
    @test domain.D == 0
    @test domain.γ_b == 1
    @test domain.ρ_liquid == 1

    set_foam!(model; D=0.03, k_H=0.001, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    @test domain.D == Float32(0.03)
    @test domain.k_H == Float32(0.001)
    stats = bubble_stats(model)
    @test stats.n == 0
    @test stats.ΣV == 0
    @test stats.mean_ratio == 0
    @test stats.max_abs_ρb == 1
    @test stats.max_Π == 0
    @test !hasproperty(stats, :clamp_hits)

    fill!(domain.flags.data, TYPE_F)
    initialize!(model)
    LatticeBoltzmann.step!(model)
    for arr in (domain.ρ.data, domain.u.data, domain.fi.data, domain.gi.data, domain.T.data,
                domain.ci.data, domain.c.data, domain.ϕ_old.data, domain.ρb.data, domain.Pi.data)
        @test all(isfinite, Array(arr))
    end
    @test all(iszero, Array(domain.ϕ_old.data))

    model2 = Model(4, 4, 4, 0.1; backend=CPU(), n_hydro=2)
    @test_throws "FOAM v1 requires n_hydro == 1 (got 2); substep scaling is not applied" LatticeBoltzmann.step!(model2)
end

# Sum of the seven populations as the next collide would load them.
# Completed steps leave domain.t = nsteps; the next substep uses isodd(t).
function _foam_macro_c(domain, n, x, y, z, t_odd::Val{odd}) where {odd}
    N = domain.N
    Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
    ci = domain.ci.data
    CType = eltype(ci)
    s = CType(ci[LatticeBoltzmann.f_index(n, 1, N)])
    for k in 1:3
        i = 2k
        cx = k == 1 ? 1 : 0
        cy = k == 2 ? 1 : 0
        cz = k == 3 ? 1 : 0
        src = LatticeBoltzmann.src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        fp, fm = LatticeBoltzmann.load_pair(ci, n, src, i, t_odd, N, CType)
        s += fp + fm
    end
    return s
end

function foam_macro_c(domain, n, x, y, z)
    # n_hydro is 1, so the next collide parity is isodd(domain.t).
    if isodd(Int(domain.t))
        return _foam_macro_c(domain, n, x, y, z, Val(true))
    else
        return _foam_macro_c(domain, n, x, y, z, Val(false))
    end
end

function foam_mean_c(domain)
    return sum(Float64, Array(domain.ci.data)) / domain.N
end

function foam_sine_amplitude(domain)
    Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
    acc = 0.0
    norm = 0.0
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        n = 1 + x + Nx * (y + Ny * z)
        s = sin(2π * x / Nx)
        acc += Float64(foam_macro_c(domain, n, x, y, z)) * s
        norm += s * s
    end
    return acc / norm
end

@testset "pure diffusion of a sine" begin
    Nx, Ny, Nz = 48, 4, 4
    D = 0.05
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    fill!(domain.T.data, 1.25f0)
    set_foam!(model; D=D, q=0, c0=0)
    initialize!(model)
    N = domain.N
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        n = 1 + x + Nx * (y + Ny * z)
        cv = Float32(sin(2π * x / Nx))
        LatticeBoltzmann.store_ceq!(domain.ci.data, n, x, y, z, cv,
                                    0.0f0, 0.0f0, 0.0f0, N, Nx, Ny, Nz, Val(false))
    end
    gi0 = copy(Array(domain.gi.data))
    @test !all(iszero, gi0)
    @test foam_sine_amplitude(domain) ≈ 1.0 atol=1.0e-5

    for _ in 1:200
        LatticeBoltzmann.step!(model)
    end
    @test Int(domain.t) == 200
    @test all(==(TYPE_F), Array(domain.flags.data))
    k = 2π / Nx
    A = exp(-D * k * k * 200)
    @test foam_sine_amplitude(domain) ≈ A rtol=0.03
    @test Array(domain.gi.data) == gi0
    @test maximum(abs, Array(domain.u.data)) < 1.0f-5
end

@testset "uniform concentration is stationary" begin
    c0 = 0.25f0
    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    fill!(domain.T.data, 1.25f0)
    set_foam!(model; D=0.05, q=0, c0=c0)
    initialize!(model)
    gi0 = copy(Array(domain.gi.data))
    @test !all(iszero, gi0)
    m0 = foam_mean_c(domain)
    @test m0 ≈ Float64(c0) atol=1.0e-5

    LatticeBoltzmann.step!(model) # even
    m1 = foam_mean_c(domain)
    @test m1 ≈ Float64(c0) atol=1.0e-5

    LatticeBoltzmann.step!(model) # odd
    m2 = foam_mean_c(domain)
    @test m2 ≈ Float64(c0) atol=1.0e-5
    @test abs(m2 - m0) < 1.0e-5
    @test Array(domain.gi.data) == gi0
end

@testset "source q on liquid only" begin
    c0 = 0.2f0
    q = 0.015f0

    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    set_foam!(model; D=0.05, q=q, c0=c0)
    initialize!(model)
    m0 = foam_mean_c(domain)
    LatticeBoltzmann.step!(model)
    @test foam_mean_c(domain) ≈ m0 + Float64(q) atol=1.0e-5
    LatticeBoltzmann.step!(model)
    @test foam_mean_c(domain) ≈ m0 + 2 * Float64(q) atol=1.0e-5

    modelI = Model(8, 8, 8, 0.1; backend=CPU())
    domainI = modelI.domains[1]
    fill!(domainI.flags.data, TYPE_I)
    set_foam!(modelI; D=0.05, q=q, c0=c0)
    initialize!(modelI)
    mi0 = foam_mean_c(domainI)
    LatticeBoltzmann.step!(modelI)
    @test foam_mean_c(domainI) ≈ mi0 atol=1.0e-5
end
