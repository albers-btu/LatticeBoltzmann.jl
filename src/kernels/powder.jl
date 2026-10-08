using KernelAbstractions

# Add an outstanding enthalpy debit back into Q. One thread per cell.
@kernel function add_qhold_kernel!(Q, qhold)
    n = @index(Global)
    @inbounds begin
        q = qhold[n]
        if q != zero(typeof(q))
            Q[n] += q
            qhold[n] = zero(typeof(q))
        end
    end
end

# One thread per parcel slot. Direction is the stored velocity over the
# step length, then renormalized, matching the host walk. Captured mass
# and enthalpy accumulate in a 2-vector so the host copy waits until the
# hydro kernels have been queued.
@kernel function walk_parcels_kernel!(
    mp, mass, Q, qhold, Tfield, fs, ρ, flags, ϕ, locks, ledger,
    px, py, pz, pvx, pvy, pvz, pm, pT, alive,
    τ_p::T, Ts::T, Λ::T, γs::T, γl::T, dist::T,
    Nx::Int, Ny::Int, Nz::Int) where {T}
    i = @index(Global)
    @inbounds begin
        if alive[i] != 0x00
            inv = dist > zero(T) ? one(T) / dist : one(T)
            dirx = pvx[i] * inv
            diry = pvy[i] * inv
            dirz = pvz[i] * inv
            nd = sqrt(dirx * dirx + diry * diry + dirz * dirz)
            if nd <= zero(T)
                alive[i] = 0x00
            else
                dirx /= nd
                diry /= nd
                dirz /= nd
                ox, oy, oz, live, dm, dE = _walk_parcel!(
                    mp, mass, Q, qhold, Tfield, fs, ρ, flags, ϕ, τ_p,
                    px[i], py[i], pz[i], dirx, diry, dirz,
                    dist, pm[i], pT[i],
                    Ts, Λ, γs, γl, Nx, Ny, Nz, locks)
                px[i] = ox
                py[i] = oy
                pz[i] = oz
                alive[i] = live ? 0x01 : 0x00
                LT = eltype(ledger)
                acc_add!(ledger, 1, LT(dE))
                acc_add!(ledger, 2, LT(dm))
            end
        end
    end
end
