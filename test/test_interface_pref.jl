using Test
using LatticeBoltzmann

@testset "interface preference" begin
    @test INTERFACE == :sharp
    @test ALLEN_CAHN == false
    @test SURFACE == true
end
