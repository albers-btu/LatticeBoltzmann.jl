struct Domain
    Nx::UInt # lattice dimension x
    Ny::UInt # lattice dimension y
    Nz::UInt # lattice dimension z

    Ox::UInt # offset x
    Oy::UInt # offset y
    Oz::UInt # offset z

    ν::Float32 # kinematic shear viscosity

    fx::Float32 # global force per volume x
    fy::Float32 # global force per volume y
    fz::Float32 # global force per volume z
end