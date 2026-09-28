"""
    set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)

Dissolved-gas parameters. Henry uses the paper, `c_H = k_H * ρ_b / 3`.
LBfoam's C++ uses `c = kh_LB * ρ_b / 4`, so `k_H = kh_LB` is 4/3 of that
interface concentration and `k_H = (3/4) * kh_LB` matches an XML run.
Their `pi_LB` is `k_Π * d_max` (`d_max = 4`), so `k_Π = pi_LB / 4`.
"""
function set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    @static if FOAM
        for domain in model.domains
            CT = typeof(domain.D)
            domain.D = CT(D)
            domain.k_H = CT(k_H)
            domain.k_Π = CT(k_Π)
            domain.q = CT(q)
            domain.V_m = CT(V_m)
            domain.γ_b = CT(γ_b)
            domain.c0 = CT(c0)
            domain.ρ_liquid = CT(ρ_liquid)
        end
        return model
    else
        throw(ArgumentError("FOAM is false"))
    end
end

"""
    bubble_stats(model) -> NamedTuple

`(n, ΣV, mean_ratio, max_abs_ρb, max_Π)`. No clamp-hit count: κ is not
in a copy of `ρb` and `Pi`.
"""
function bubble_stats(model)
    @static if FOAM
        n = 0
        ΣV = 0.0
        mean_ratio = 0.0
        max_abs_ρb = 0.0
        max_Π = -Inf
        for domain in model.domains
            ρb = Array(domain.ρb.data)
            Pi = Array(domain.Pi.data)
            isempty(ρb) || (max_abs_ρb = max(max_abs_ρb, Float64(maximum(abs, ρb))))
            isempty(Pi) || (max_Π = max(max_Π, Float64(maximum(Pi))))
        end
        max_Π == -Inf && (max_Π = 0.0)
        return (; n, ΣV, mean_ratio, max_abs_ρb, max_Π)
    else
        throw(ArgumentError("FOAM is false"))
    end
end

@static if FOAM
    function foam_host!(model, domain)
        return nothing
    end
end
