@inline function wrap_coord(x, dx, N)
    ifelse(dx == 0, x,
        ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                       ifelse(x == 0, N - 1, x - 1)))
end

@inline function src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
    wrap_coord(x, cx, Nx) + 
    wrap_coord(y, cy, Ny) * Nx + 
    wrap_coord(z, cz, Nz) * Nx * Ny + 1
end

# EsotericPull. store(P) is what the next load(!P) pulls. A second step on the same
# parity reads that store with + and − swapped, so the populations do not stream.
@inline load_pair(fi, n, src, i, ::Val{true}, N, ::Type{CType}) where {CType} =
    (
        CType(fi[f_index(n, i, N)]),        # f₊
        CType(fi[f_index(src, i + 1, N)])   # f₋
    )
@inline load_pair(fi, n, src, i, ::Val{false}, N, ::Type{CType}) where {CType} =
    (
        CType(fi[f_index(n, i + 1, N)]),    # f₊
        CType(fi[f_index(src, i, N)])       # f₋
    )

@inline function store_pair!(fi, n, src, i, f_plus, f_minus, ::Val{true}, N)
    fi[f_index(src, i + 1, N)] = eltype(fi)(f_plus)     # f₊
    fi[f_index(n, i, N)]       = eltype(fi)(f_minus)    # f₋
    return nothing
end
@inline function store_pair!(fi, n, src, i, f_plus, f_minus, ::Val{false}, N)
    fi[f_index(src, i, N)]     = eltype(fi)(f_plus)     # f₊
    fi[f_index(n, i + 1, N)]   = eltype(fi)(f_minus)    # f₋
    return nothing
end

@inline load_outgoing_pair(fi, n, src, i, ::Val{true}, N, ::Type{CType}) where {CType} =
    (
        CType(fi[f_index(src, i, N)]),      # f₊
        CType(fi[f_index(n, i + 1, N)])     # f₋
    )
@inline load_outgoing_pair(fi, n, src, i, ::Val{false}, N, ::Type{CType}) where {CType} =
    (
        CType(fi[f_index(src, i + 1, N)]),  # f₊
        CType(fi[f_index(n, i, N)])         # f₋
    )

@inline function load_bb_pair(fi, flags, n, src_p, src_m, i, t_odd::Val, N, ::Type{CType}) where {CType}
    fp, fm = load_pair(fi, n, src_p, i, t_odd, N, CType)
    solid_p = (flags[src_p] & TYPE_BO) == TYPE_S
    solid_m = (flags[src_m] & TYPE_BO) == TYPE_S
    if solid_p | solid_m
        fp_out, fm_out = load_outgoing_pair(fi, n, src_p, i, t_odd, N, CType)
        solid_m && (fp = fm_out)
        solid_p && (fm = fp_out)
    end
    return fp, fm
end

@inline function store_reconstructed_pair!(fi, n, src, i, f_plus, f_minus, gas_plus, gas_minus, ::Val{true}, N)
    gas_minus && (fi[f_index(n, i, N)]       = eltype(fi)(f_minus))
    gas_plus  && (fi[f_index(src, i + 1, N)] = eltype(fi)(f_plus))
    return nothing
end
@inline function store_reconstructed_pair!(fi, n, src, i, f_plus, f_minus, gas_plus, gas_minus, ::Val{false}, N)
    gas_minus && (fi[f_index(n, i + 1, N)] = eltype(fi)(f_minus))
    gas_plus  && (fi[f_index(src, i, N)]   = eltype(fi)(f_plus))
    return nothing
end