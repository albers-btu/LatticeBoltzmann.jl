struct Domain
    Nx::Int # lattice dimension x
    Ny::Int # lattice dimension y
    Nz::Int # lattice dimension z
    Dx::Int # lattice domain x
    Dy::Int # lattice domain y
    Dz::Int # lattice domain z
    τ::Float32 # time step

    ν::Float32 # kinematic shear viscosity
    fx::Float32 # global force per volume x
    fy::Float32 # global force per volume y
    fz::Float32 # global force per volume z
    σ::Float32 # surface tension coefficient
    α::Float32 # thermal diffusion coefficient
    β::Float32 # thermal expansion coefficient
    T_avg::Float32 # average temperature
end

Domain(Nx, Ny, Nz, ν) = Domain(
    Int(Nx), 
    Int(Ny), 
    Int(Nz), 
    1, 1, 1, 1.0f0,
    Float32(ν), 
    0.0f0, 0.0f0, 0.0f0, 
    0.0f0, 1.0f0, 1.0f0, 
    1.0f0
)