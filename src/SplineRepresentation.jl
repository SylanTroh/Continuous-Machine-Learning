"""
Piecewise linear representation of f: [0,1] -> T, stored as values at N evenly spaced
control points. Hat-function basis, so basis_weights(f, t) has at most two nonzero entries.
"""
module SplineRepresentation

using LinearAlgebra
using Random
using ..AbstractRepresentation
using ..LinearInterp
import ..AbstractRepresentation: basis_weights, device_basis_weights, nodes, reconstruct, random_like, _norm_integrand, _dot_integrand

export Spline, RandomSpline, trapz

#Trapezoid rule on [0,1]
_trapz(values, N) = sum(values[1:end-1] + values[2:end]) / (2(N - 1))

"""
    Spline{T} <: AbstractFunctionRepresentation{T}

Piecewise linear spline.
f: [0,1] → T.
"""
mutable struct Spline{T} <: AbstractFunctionRepresentation{T}
    vals::Vector{T}
    N::Int

    function Spline(ys::AbstractVector{T}) where T<:Union{Real,AbstractArray}
        isempty(ys) && throw(ArgumentError("Spline control points cannot be empty"))
        new{T}(Vector(ys), length(ys))
    end
end

"""
    RandomSpline(n, [T=Float64])

Random scalar spline with n control points.
"""
RandomSpline(n::Int, ::Type{T}=Float64) where T<:Real = Spline(randn(T, n))

"""
    RandomSpline(n, dimension, [T=Float64])

Random spline with n control points, each an array of given dimension.
"""
RandomSpline(n::Int, dimension::Tuple, ::Type{T}=Float64) where T<:Real =
    Spline([randn(T, dimension) for _ in 1:n])

"""
    (f::Spline)(t::Real)

Evaluate spline at t in [0,1].
"""
(f::Spline)(t::Real) = interpolate_samples(f.vals, t)

LinearAlgebra.norm(f::Spline) = sqrt(_trapz(_norm_integrand(f), f.N))

function LinearAlgebra.dot(f::Spline, g::Spline)
    length(f) != length(g) && throw(DimensionMismatch("Spline lengths must match"))
    return _trapz(_dot_integrand(f, g), f.N)
end

trapz(f::Spline) = _trapz(f.vals, f.N)

function basis_weights(f::Spline, t::Real)
    N = f.N
    w = zeros(Float64, N)
    t = clamp(t, 0.0, 1.0)
    scaled = t * (N - 1) + 1
    i = floor(Int, scaled)
    if i >= N
        w[N] = 1.0
    elseif i < 1
        w[1] = 1.0
    else
        α = scaled - i
        w[i] = 1 - α
        w[i+1] = α
    end
    return w
end

#Branchless basis_weights: evaluates as a broadcast on x's storage (no host→device upload).
function device_basis_weights(f::Spline, t::Real, x::AbstractArray)
    T = eltype(x)
    N = f.N
    k = similar(x, T, N)
    k .= 1:N
    scaled = T(clamp(t, 0, 1) * (N - 1) + 1)
    return max.(zero(T), one(T) .- abs.(scaled .- k))
end

nodes(f::Spline) = range(0.0, 1.0, length=f.N)

reconstruct(f::Spline, new_vals) = Spline(new_vals)

random_like(::Type{Spline}, n::Int, ::Type{T}=Float64) where T<:Real = RandomSpline(n, T)
random_like(::Type{Spline}, n::Int, dims::Tuple, ::Type{T}=Float64) where T<:Real =
    RandomSpline(n, dims, T)

end
