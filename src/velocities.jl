using StaticArrays

const VELOCITIES = Dict(
    :D2Q9 => SVector{3, Int}[
        SVector( 0,  0,  0),
        SVector( 1,  0,  0),
        SVector(-1,  0,  0),
        SVector( 0,  1,  0),
        SVector( 0, -1,  0),
        SVector( 1,  1,  0),
        SVector(-1, -1,  0),
        SVector( 1, -1,  0),
        SVector(-1,  1,  0)
    ],
    :D3Q7 => SVector{3, Int}[
        SVector( 0,  0,  0),
        SVector( 1,  0,  0),
        SVector(-1,  0,  0),
        SVector( 0,  1,  0),
        SVector( 0, -1,  0),
        SVector( 0,  0,  1),
        SVector( 0,  0, -1)
    ],
    :D3Q19 => SVector{3, Int}[
        SVector( 0,  0,  0 ),
        SVector( 1,  0,  0 ),
        SVector(-1,  0,  0 ),
        SVector( 0,  1,  0 ),
        SVector( 0, -1,  0 ),
        SVector( 0,  0,  1 ),
        SVector( 0,  0, -1 ),
        SVector( 1,  1,  0 ),
        SVector(-1, -1,  0 ),
        SVector( 1,  0,  1 ),
        SVector(-1,  0, -1 ),
        SVector( 0,  1,  1 ),
        SVector( 0, -1, -1 ),
        SVector( 1, -1,  0 ),
        SVector(-1,  1,  0 ),
        SVector( 1,  0, -1 ),
        SVector(-1,  0,  1 ),
        SVector( 0,  1, -1 ),
        SVector( 0, -1,  1 )
    ],
    :D3Q27 => SVector{3, Int}[
        SVector( 0,  0,  0 ),
        SVector( 1,  0,  0 ),
        SVector(-1,  0,  0 ),
        SVector( 0,  1,  0 ),
        SVector( 0, -1,  0 ),
        SVector( 0,  0,  1 ),
        SVector( 0,  0, -1 ),
        SVector( 1,  1,  0 ),
        SVector(-1, -1,  0 ),
        SVector( 1,  0,  1 ),
        SVector(-1,  0, -1 ),
        SVector( 0,  1,  1 ),
        SVector( 0, -1, -1 ),
        SVector( 1, -1,  0 ),
        SVector(-1,  1,  0 ),
        SVector( 1,  0, -1 ),
        SVector(-1,  0,  1 ),
        SVector( 0,  1, -1 ),
        SVector( 0, -1,  1 ),
        SVector( 1,  1,  1 ),
        SVector(-1, -1, -1 ),
        SVector( 1,  1, -1 ),
        SVector(-1, -1,  1 ),
        SVector( 1, -1,  1 ),
        SVector(-1,  1, -1 ),
        SVector(-1,  1,  1 ),
        SVector( 1, -1, -1 )
    ]
)

velocities(scheme::Symbol) = Tuple(VELOCITIES[scheme])