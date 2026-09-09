using Test
using LatticeBoltzmann

@testset "Different model setups" begin
    model = Model(256, 256, 256, 1.0)

	@test length(model.domains) == 1
	@test length(model.domains[1].ρ) == 256^3

	@test model.ρ[1] == Float32(1.0)
	@test model.u[1, 1] == Float32(0.0)
	@test model.flags[1] == 0x00

    @test model.domains[1].ρ === model.ρ.buffers[1]
    @test model.domains[1].u === model.u.buffers[1]
	@test model.domains[1].flags === model.flags.buffers[1]

	model.flags[1] = TYPE_S
	@test model.flags[1] == TYPE_S
end