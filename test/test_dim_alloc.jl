using Test
using LatticeBoltzmann

@testset "dimension allocation" begin
    @static if DIM == 3
        model = Model(8, 8, 8, 1.0)
        @test length(model.weights) == 19
        @test length(model.domains[1].gi) == 8^3 * 7

        slab = Model(8, 8, 1, 1.0)
        @test length(slab.weights) == 19
        @test length(slab.domains[1].gi) == 8 * 8 * 1 * 7
    end

    @static if DIM == 2
        model = Model(8, 8, 1, 1.0)
        N = 8 * 8
        @test length(model.weights) == 9
        @test length(model.domains[1].gi) == N * 5
        @test size(model.domains[1].u) == (N, 3)

        @test_throws ArgumentError Model(8, 8, 4, 1.0)
        @test_throws ArgumentError Model(8, 8, 1, 1.0; scheme=:D3Q27)
        explicit = Model(8, 8, 1, 1.0; scheme=:D2Q9)
        @test length(explicit.weights) == 9
    end
end
