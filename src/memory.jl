using Adapt, StaticArrays, CUDA, KernelAbstractions

struct Memory{T,A<:AbstractArray{T}}
    data::A # host or device
end

Adapt.adapt_structure(to, m::Memory) = Memory(adapt(to, m.data))