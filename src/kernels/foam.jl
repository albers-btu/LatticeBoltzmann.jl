using KernelAbstractions

@static if FOAM

@inline function concentration_body!(::Val{odd}, ::Int) where {odd}
    return nothing
end

@kernel function concentration_even_kernel!()
    n = @index(Global)
    @inbounds concentration_body!(Val(false), Int(n))
end

@kernel function concentration_odd_kernel!()
    n = @index(Global)
    @inbounds concentration_body!(Val(true), Int(n))
end

@inline function disjoining_body!(::Int)
    return nothing
end

@kernel function disjoining_kernel!()
    n = @index(Global)
    @inbounds disjoining_body!(Int(n))
end

@inline function ϕ_correction_body!(::Int)
    return nothing
end

@kernel function ϕ_correction_kernel!()
    n = @index(Global)
    @inbounds ϕ_correction_body!(Int(n))
end

end
