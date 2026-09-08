using Test
using LatticeBoltzmann

@testset "Different domain sizes" begin
    domain = Domain(256, 256, 256, 1.0)
    @test 1==1
end