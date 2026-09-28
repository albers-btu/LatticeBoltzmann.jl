# HDF5 powder-bed export for LatticeBoltzmann.jl

Write one HDF5 file per bed with **HDF5.jl**. Do not transpose the array. HDF5.jl stores a Julia array so that a Julia reader gets the same indices back. A Python/h5py reader will see the dimensions reversed; that is expected.

## Datasets

### `phi` (required)

- Julia type: `Array{Float32,3}`
- Size: `(Nx, Ny, Nz)`
- Index: `phi[x, y, z]` with `x = 1:Nx`, `y = 1:Ny`, `z = 1:Nz`
- Value: metal volume fraction in that cell, in `[0, 1]`
  - `0` = no metal
  - `1` = cell full of metal
  - a particle that only covers part of a cell writes that fraction
  - overlapping particles are summed and then clamped to `1`
- Do not threshold to `{0,1}`

### `substrate` (optional)

- Julia type: `Array{UInt8,3}`, same size and index as `phi`
- `1` = base plate, `0` = not plate
- Omit the dataset if there is no plate
- Do not also put the plate into `phi`

## Attributes on the file root

All lengths in **metres**.

| name | Julia type | meaning |
|---|---|---|
| `dx`, `dy`, `dz` | `Float64` | cell edge. Equal for a cubic grid. |
| `origin` | `Vector{Float64}` length 3 | **minimum corner** of cell `(1,1,1)`, not the cell centre |
| `index_order` | `String` | `"x_fastest"` |
| `cell_center` | `String` | `"integer"` |
| `quantity` | `String` | `"metal_volume_fraction"` |

## Where a cell is

Cell `(x, y, z)` (1-based) has its centre at

```text
origin .+ ( (x - 0.5) * dx,  (y - 0.5) * dy,  (z - 0.5) * dz )
```

In the lattice coordinate used by this solver (`floor(p + 0.5)` picks the cell), that centre is the integer point `(x, y, z)`.

A DEM particle at physical position `r` (metres) belongs to cell

```julia
x = floor(Int, (r[1] - origin[1]) / dx) + 1
y = floor(Int, (r[2] - origin[2]) / dy) + 1
z = floor(Int, (r[3] - origin[3]) / dz) + 1
```

and only if `1 ≤ x ≤ Nx` and the same for `y`, `z`.

## What this solver will do with the file

- `phi == 0` and not substrate → empty. `initialize!` turns that into gas.
- `phi > 0` → unmelted powder mass `mp = phi` (lattice density is 1). Not a bounce-back wall.
- `substrate == 1` → `TYPE_F`, solid fraction `fs = 1`, temperature below the solidus. This metal can melt. It is not `TYPE_S`.
- There is no `flags` dataset. This solver chooses the flags.

## Writer

```julia
using HDF5

function write_powder_bed(path::AbstractString,
                          phi::AbstractArray{<:Real,3},
                          dx::Real, dy::Real, dz::Real,
                          origin::NTuple{3,Real};
                          substrate::Union{Nothing,AbstractArray{UInt8,3}}=nothing)
    phi32 = Float32.(clamp.(phi, 0, 1))
    h5open(path, "w") do f
        f["phi"] = phi32
        if substrate !== nothing
            size(substrate) == size(phi32) ||
                error("substrate size $(size(substrate)) != phi size $(size(phi32))")
            f["substrate"] = substrate
        end
        attributes(f)["dx"] = Float64(dx)
        attributes(f)["dy"] = Float64(dy)
        attributes(f)["dz"] = Float64(dz)
        attributes(f)["origin"] = Float64[origin...]
        attributes(f)["index_order"] = "x_fastest"
        attributes(f)["cell_center"] = "integer"
        attributes(f)["quantity"] = "metal_volume_fraction"
    end
    return nothing
end
```

Do not call `permutedims` before writing. `phi[x,y,z]` in the array above is cell `(x,y,z)`.

## Check before sending a file

```julia
using HDF5
h5open("bed.h5", "r") do f
    phi = f["phi"][:, :, :]
    @assert eltype(phi) == Float32
    @assert ndims(phi) == 3
    @assert attributes(f)["index_order"] == "x_fastest"
    @assert attributes(f)["quantity"] == "metal_volume_fraction"
    @assert all(0 .<= phi .<= 1)
    # one particle whose centre is the centre of cell (x,y,z)
    # must satisfy phi[x, y, z] > 0
end
```

A file that looks right in h5py with shape `(Nx, Ny, Nz)` and `phi[0,0,0]` as the first x-index is **wrong**. h5py should report shape `(Nz, Ny, Nx)`.
