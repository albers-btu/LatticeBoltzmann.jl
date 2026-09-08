using Test
using LatticeBoltzmann

@testset "Different model setups" begin
    model = Model(256, 256, 256, 1.0)

    @test 1==1
end