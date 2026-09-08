using KernelAbstractions
using Test
using BenchmarkTools
using CUDA

const n::Int = 1_000_000

@kernel function vector_add!(a, b, c)
    i = @index(Global)
    @inbounds c[i] = a[i] + b[i]
end

function wait_if_needed(event)
    event === nothing || wait(event)
end

@testset "CPU Kernel functionality" begin
    backend = CPU()

    a = rand(Float32, n)
    b = rand(Float32, n)
    c = similar(a)

    event = vector_add!(backend, n)(a, b, c; ndrange=n)
    wait_if_needed(event)

    trial = @benchmark begin
        event = vector_add!($backend, $n)($a, $b, $c; ndrange=$n)
        wait_if_needed(event)
    end

    display(trial)

    @test c == a .+ b
end

@testset "GPU kernel functionality" begin
    if CUDA.functional()
        backend = CUDABackend()

        a = CUDA.rand(Float32, n)
        b = CUDA.rand(Float32, n)
        c = similar(a)

        event = vector_add!(backend, 256)(a, b, c; ndrange=n)
        wait_if_needed(event)
        CUDA.synchronize()

        trial = @benchmark begin
            event = vector_add!($backend, 256)($a, $b, $c; ndrange=$n)
            wait_if_needed(event)
            CUDA.synchronize()
        end
        display(trial)

        @test Array(c) == Array(a) .+ Array(b)
    else
        @info "Skipping GPU tests: CUDA is not functional"
    end
end