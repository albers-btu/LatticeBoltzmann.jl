using KernelAbstractions

# One thread per ray. flags and ϕ stay on the device. Heat scatters with
# a float atomic. The bundle offsets are uploaded once; the origin is the
# current beam position passed as scalars.
@kernel function deposit_rays_kernel!(
    Q, flags, ϕ, ox, oy, Pray,
    x::T, y::T, z::T, dx::T, dy::T, dz::T,
    n_re::T, n_im::T, max_bounce::Int, skin::Int, qfac::T, scale::T,
    Nx::Int, Ny::Int, Nz::Int) where {T}
    rid = @index(Global)
    @inbounds begin
        rx, ry, rz = ray_origin(x, y, z, dx, dy, dz, ox[rid], oy[rid])
        _walk_laser_ray!(
            Q, flags, ϕ, rx, ry, rz, dx, dy, dz, Pray[rid] * scale,
            n_re, n_im, max_bounce, skin, qfac, Nx, Ny, Nz, nothing)
    end
end
