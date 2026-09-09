using Adapt, StaticArrays, CUDA, KernelAbstractions

@inline f_index(n, i, N) = n + (i - 1) * N

struct Memory{T, A<:AbstractArray{T}}
    data::A # host or device
end

Adapt.adapt_structure(to, m::Memory) = Memory(adapt(to, m.data))

Base.size(m::Memory) = size(m.data)
Base.length(m::Memory) = length(m.data)
Base.eltype(::Memory{T}) where {T} = T
Base.getindex(m::Memory, i...) = m.data[i...]
Base.setindex!(m::Memory, v, i...) = (m.data[i...] = v)

struct MemoryContainer{T, A<:AbstractArray{T}}
    buffers::Vector{Memory{T, A}}

    Nx::UInt
    Ny::UInt
    Nz::UInt

    Dx::UInt
    Dy::UInt
    Dz::UInt

    name::String
end

function attach(buffers::Vector{<:Memory{T, A}}, Nx, Ny, Nz, Dx, Dy, Dz, name::String) where
    {T, A}
        length(buffers) == Int(Dx * Dy * Dz) || throw(ArgumentError("expected $(Int(Dx*Dy*Dz)) domain buffers for $(name), got $(length(buffers))"))

    MemoryContainer{T, A}(
        buffers, 
        UInt(Nx), UInt(Ny), UInt(Nz), 
        UInt(Dx), UInt(Dy), UInt(Dz), 
        name
    )
end

function _local_index(c::MemoryContainer, n::Integer)
    D = length(c.buffers)
    if D == 1
        return 1, n
    end

    n0 = n - 1
    Nx, Ny, Nz = Int(c.Nx), Int(c.Ny), Int(c.Nz)
    Dx, Dy, Dz = Int(c.Dx), Int(c.Dy), Int(c.Dz)
    NxNy = Nx * Ny
    t = n0 % NxNy
    x = t % Nx
    y = t ÷ Nx
    z = n0 ÷ NxNy

    NxDx, NyDy, NzDz = Nx ÷ Dx, Ny ÷ Dy, Nz ÷ Dz
    Hx, Hy, Hz = Int(Dx > 1), Int(Dy > 1), Int(Dz > 1)
    px, py, pz = x % NxDx, y % NyDy, z % NzDz
    dx, dy, dz = x ÷ NxDx, y ÷ NyDy, z ÷ NzDz
    domain0 = dx + (dy + dz * Dy) * Dx
    local_Nx = NxDx + 2 * Hx
    local_Ny = NyDy + 2 * Hy
    local_i0 = (px + Hx) + ((py + Hy) + (pz + Hz) * local_Ny) * local_Nx

    return domain0 + 1, local_i0 + 1
end

function Base.getindex(c::MemoryContainer, n::Integer)
    d, i = _local_index(c, n)
    c.buffers[d][i]
end

function Base.setindex!(c::MemoryContainer, v, n::Integer)
    d, i = _local_index(c, n)
    c.buffers[d][i] = v
end

function Base.getindex(c::MemoryContainer, n::Integer, component::Integer)
    d, i = _local_index(c, n)
    data = c.buffers[d].data
    if ndims(data) == 2
        return data[i, component] # ndims(u) = 2 ... (N, 3)
    else
        Nloc = _local_N(c)
        return data[f_index(i, component, Nloc)]
    end
end

function Base.setindex!(c::MemoryContainer, v, n::Integer, component::Integer)
    d, i = _local_index(c, n)
    data = c.buffers[d].data
    if ndims(data) == 2
        data[i, component] = v
    else
        Nloc = _local_N(c)
        data[f_index(i, component, Nloc)] = v
    end
    return v
end

Base.length(c::MemoryContainer) = Int(c.Nx) * Int(c.Ny) * Int(c.Nz)

_local_N(c::MemoryContainer) = Int(c.Nx ÷ c.Dx + 2 * (c.Dx > 1)) *
                               Int(c.Ny ÷ c.Dy + 2 * (c.Dy > 1)) *
                               Int(c.Nz ÷ c.Dz + 2 * (c.Dz > 1))

                            