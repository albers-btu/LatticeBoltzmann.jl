using Test
using LatticeBoltzmann
using KernelAbstractions

@inline function lbm_n(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

@testset "TEMPERATURE Dirichlet conduction" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 8, 8, 16
    α = 0.05f0
    model = Model(Nx, Ny, Nz, 0.05; α = α, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = ones(Float32, Nx * Ny * Nz)
    T_hot, T_cold = 1.5f0, 0.5f0
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_T
            Th[n] = T_hot
        elseif z == Nz - 1
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    run!(model, 800)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    mid = lbm_n(4, 4, Nz ÷ 2, Nx, Ny)
    @test T[mid] > T_cold + 0.15f0
    @test T[mid] < T_hot - 0.15f0
    @test isfinite(T[mid])
end

@testset "TEMPERATURE volumetric Q heats uniformly" begin
    Nx = Ny = Nz = 8
    Qv = 2.0f-4
    nsteps = 40
    model = Model(Nx, Ny, Nz, 0.05; α = 0.05f0, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    fill!(model.domains[1].flags.data, TYPE_F)
    fill!(model.domains[1].Q.data, Qv)
    LatticeBoltzmann.initialize!(model)
    T0 = Array(model.domains[1].T.data)
    run!(model, nsteps)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    ΔT = sum(T - T0) / length(T)
    @test isapprox(ΔT, nsteps * Qv; rtol=0.12)
    @test all(isfinite, T)
end

@testset "TEMPERATURE Neumann flux vs Fourier" begin
    Nx, Ny, Nz = 6, 6, 16
    α = 0.2f0
    q_in = 4.0f-3
    T_cold = 1.0f0
    model = Model(Nx, Ny, Nz, 0.2; α = α, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(T_cold, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_H
            Qh[n] = q_in
        elseif z == Nz - 1
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].Q.data, Qh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 2500)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    zH, zC = 2, Nz - 1
    L = Float32(zC - zH)
    nH = lbm_n(3, 3, zH, Nx, Ny)
    nC = lbm_n(3, 3, zC, Nx, Ny)
    nM = lbm_n(3, 3, (zH + zC) ÷ 2, Nx, Ny)
    T_H, T_M, T_C = T[nH], T[nM], T[nC]
    @test T_H > T_C + 0.05f0
    @test isapprox(T_H, T_cold + q_in / k * L; rtol=0.2)
    @test isapprox(T_M, T_cold + q_in / k * (L / 2); rtol=0.25)
    @test isfinite(T_H) && isfinite(T_M)
end

@testset "TEMPERATURE Robin large h approaches T∞" begin
    Nx, Ny, Nz = 6, 6, 12
    α = 0.2f0
    T∞, T_cold = 1.6f0, 0.8f0
    hconv = 8.0f0
    model = Model(Nx, Ny, Nz, 0.2; α = α, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Float32((T∞ + T_cold) / 2), Nx * Ny * Nz)
    hh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_H
            Th[n] = T∞
            hh[n] = hconv
        elseif z == Nz - 1
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].h.data, hh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 2000)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    zH, zC = 2, Nz - 1
    L = Float32(zC - zH)
    Bi = hconv / k
    A = (T_cold + Bi * L * T∞) / (1 + Bi * L)
    nH = lbm_n(3, 3, zH, Nx, Ny)
    nF = lbm_n(3, 3, zH + 1, Nx, Ny)
    @test isapprox(T[nF], A + (T_cold - A) / L; rtol=0.2)
    @test T[nF] > T_cold
    @test T[nF] < T∞
end

@testset "SURFACE+T insulated liquid layer" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 8, 8, 20
    α = 0.2f0
    Qv = 2.0f-5
    T_b = 1.0f0
    Hfill = 12
    model = Model(Nx, Ny, Nz, 0.2; α = α, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(T_b, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_T
            Th[n] = T_b
            Qh[n] = Qv
        elseif z <= Hfill
            host[n] = TYPE_F
            Qh[n] = Qv
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].Q.data, Qh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 4000)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    flags = Array(model.domains[1].flags.data)
    zb = 2
    zmid = (zb + Hfill) ÷ 2
    nM = lbm_n(4, 4, zmid, Nx, Ny)
    nG = lbm_n(4, 4, Nz - 2, Nx, Ny)
    H = Float32(Hfill - zb)
    zstar = Float32(zmid - zb)
    Tan = T_b + (Qv / k) * (H * zstar - zstar^2 / 2)
    @test (flags[nM] & TYPE_F) == TYPE_F || (flags[nM] & TYPE_I) == TYPE_I
    @test (flags[nG] & TYPE_G) == TYPE_G
    @test T[nM] > T_b
    @test isapprox(T[nM], Tan; rtol=0.35)
    @test isfinite(T[nM])
end
