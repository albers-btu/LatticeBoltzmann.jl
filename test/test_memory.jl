import LatticeBoltzmann as LB
using StaticArrays, CUDA, Adapt

@testset "Move memory between backends" begin
    # Host SVector from StaticArrays
    sv = SVector{4, Float32}(1, 2, 3, 4)
    @test sv isa StaticArray

    # Host Memory with MVector inside
    host = LB.Memory(MVector{4, Float32}(1, 2, 3, 4))
    @test typeof(host.data) == MVector{4, Float32}

    # Device CuArray
    if CUDA.functional()
        dev = adapt(CuArray, host)

        @test typeof(dev.data) == CuArray{Float32, 1, CUDACore.DeviceMemory}
        @test Array(dev.data) == host.data # test round-trip
    end
end