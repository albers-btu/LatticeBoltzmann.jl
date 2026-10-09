using Test
using LatticeBoltzmann
using KernelAbstractions

@testset "Allen–Cahn allocation" begin
    backend = CPU()
    @static if DIM == 2
        Nz = 1
        nvel = 9
    else
        Nz = 8
        nvel = 19
    end
    N = 8 * 8 * Nz
    model = Model(8, 8, Nz, 0.05; backend=backend, workgroup=64)
    domain = model.domains[1]
    @test length(domain.hi) == N * nvel
    @test length(domain.phi) == N
    @test domain.W == Float32(4)
    @test domain.Mphi == Float32(0.05)
    @test domain.rho_a == Float32(1)
    @test domain.rho_b == Float32(1)
    @test domain.nu_a == domain.ν
    @test domain.nu_b == domain.ν

    custom = Model(8, 8, Nz, 0.05; backend=backend, workgroup=64,
                   W=6, Mphi=0.1, rho_a=1.5, rho_b=2, nu_a=0.02, nu_b=0.03)
    cd = custom.domains[1]
    @test cd.W == Float32(6)
    @test cd.Mphi == Float32(0.1)
    @test cd.rho_a == Float32(1.5)
    @test cd.rho_b == Float32(2)
    @test cd.nu_a == Float32(0.02)
    @test cd.nu_b == Float32(0.03)

    @test_throws ArgumentError Model(8, 8, Nz, 0.05; backend=backend, workgroup=64, n_hydro=2)
end
