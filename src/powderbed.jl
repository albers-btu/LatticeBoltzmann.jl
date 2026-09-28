using HDF5

struct PowderBed
    phi::Array{Float32,3}                          # phi[x,y,z], metal volume fraction
    substrate::Union{Nothing,Array{UInt8,3}}       # 1 = base plate
    dx::Float64
    dy::Float64
    dz::Float64
    origin::NTuple{3,Float64}                      # min corner of cell (1,1,1), metres
end

function read_powder_bed(path::AbstractString)
    isfile(path) || throw(ArgumentError("powder bed file not found: $path"))
    h5open(path, "r") do f
        haskey(f, "phi") || throw(ArgumentError("$path has no dataset phi"))
        phi = Float32.(f["phi"][:, :, :])
        ndims(phi) == 3 || throw(ArgumentError("phi must be 3D, got $(ndims(phi))"))
        attrs = HDF5.attributes(f)
        order = String(read(attrs["index_order"]))
        order == "x_fastest" || throw(ArgumentError("index_order=$order, expected x_fastest"))
        String(read(attrs["quantity"])) == "metal_volume_fraction" ||
            throw(ArgumentError("quantity must be metal_volume_fraction"))
        sub = haskey(f, "substrate") ? UInt8.(f["substrate"][:, :, :]) : nothing
        if sub !== nothing && size(sub) != size(phi)
            throw(ArgumentError("substrate size $(size(sub)) != phi size $(size(phi))"))
        end
        origin = Float64.(read(attrs["origin"]))
        length(origin) == 3 || throw(ArgumentError("origin must have length 3"))
        return PowderBed(phi, sub,
                         Float64(read(attrs["dx"])),
                         Float64(read(attrs["dy"])),
                         Float64(read(attrs["dz"])),
                         (origin[1], origin[2], origin[3]))
    end
end

# Top-most 1-based file layer that contains substrate. 0 if there is none.
function substrate_top(bed::PowderBed)
    bed.substrate === nothing && return 0
    Nx, Ny, Nz = size(bed.phi)
    for z in Nz:-1:1
        any(>(0), @view bed.substrate[:, :, z]) && return z
    end
    return 0
end

# Paint the bed into a domain that already has its walls set.
# File cell (x,y,z) lands on domain cell (x,y,z0+z-1).
# Substrate and any cell with metal become TYPE_F (meltable), not TYPE_S.
# Returns the domain z of the top substrate cell (0 if there is no plate).
function paint_powder_bed!(flags::AbstractVector{UInt8}, T::AbstractVector, fs::AbstractVector,
                           bed::PowderBed, T_init; z0::Int=2)
    Nx, Ny, Nz = size(bed.phi)
    Hfill = 0
    zsub = substrate_top(bed)
    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        zd = z0 + z - 1
        n = x + (y - 1) * Nx + (zd - 1) * Nx * Ny
        plate = bed.substrate !== nothing && bed.substrate[x, y, z] != 0x00
        metal = bed.phi[x, y, z] > 0
        if plate || metal
            flags[n] = TYPE_F
            T[n] = T_init
            fs[n] = 1
        end
        plate && z == zsub && (Hfill = zd)
    end
    return Hfill
end
