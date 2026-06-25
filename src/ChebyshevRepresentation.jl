"""
Polynomial representation of f: [0,1] -> T.
Stores values at N Chebyshev nodes (second kind, rescaled to [0,1]).
Barycentric Lagrange interpolation to evaluate.

References:
- [BT04] Berrut & Trefethen, "Barycentric Lagrange Interpolation", SIAM Review 46(3), 2004.
- [Tre00] Trefethen, "Spectral Methods in MATLAB", SIAM, 2000.
"""
module ChebyshevRepresentation

using LinearAlgebra
using Random
using ..AbstractRepresentation
import ..AbstractRepresentation: basis_weights, nodes, reconstruct, random_like, _norm_integrand, _dot_integrand

export ChebPoly, RandomChebPoly

#Chebyshev points of the second kind, rescaled to [0,1]
# [BT04, section 5]
_cheb_nodes(N::Int) = [0.5 * (1 - cos(π * j / (N - 1))) for j in 0:(N-1)]

#Barycentric weights for second-kind Chebyshev nodes
# [BT04, eq. 5.4]
function _cheb_bary_weights(N::Int)
    w = [(-1.0)^j for j in 0:(N-1)]
    w[1] *= 0.5
    w[end] *= 0.5
    return w
end

# Clenshaw-Curtis quadrature weights on [-1,1]
# port of clencurt.m [Tre00]
function _clenshaw_curtis_weights(N::Int)
    n = N - 1
    n == 0 && return [2.0]
    θ = [π * k / n for k in 0:n]
    w = zeros(N)
    if iseven(n)
        w[1] = 1.0 / (n^2 - 1)
        w[N] = w[1]
        for k in 2:N-1
            v = 1.0
            for j in 1:div(n, 2)-1
                v -= 2 * cos(2j * θ[k]) / (4j^2 - 1)
            end
            v -= cos(n * θ[k]) / (n^2 - 1)
            w[k] = 2v / n
        end
    else
        w[1] = 1.0 / n^2
        w[N] = w[1]
        for k in 2:N-1
            v = 1.0
            for j in 1:div(n - 1, 2)
                v -= 2 * cos(2j * θ[k]) / (4j^2 - 1)
            end
            w[k] = 2v / n
        end
    end
    return w
end

#Rescale quadrature weights from [-1,1] to [0,1]
_cheb_basis_integrals(N::Int) = _clenshaw_curtis_weights(N) ./ 2

"""
    ChebPoly{T} <: AbstractFunctionRepresentation{T}

Polynomial interpolation of values on N Chebyshev nodes.
f: [0,1] → T
"""
mutable struct ChebPoly{T} <: AbstractFunctionRepresentation{T}
    vals::Vector{T}
    N::Int
    nodes::Vector{Float64}
    bary::Vector{Float64}
    basis_integrals::Vector{Float64}
end

function ChebPoly(ys::AbstractVector{T}) where T<:Union{Real,AbstractArray}
    N = length(ys)
    N >= 2 || throw(ArgumentError("ChebPoly needs at least 2 values"))
    return ChebPoly{T}(Vector(ys), N, _cheb_nodes(N), _cheb_bary_weights(N),
        _cheb_basis_integrals(N))
end

"""
    RandomChebPoly(n, [T=Float64])

Random scalar polynomial with n node values.
"""
RandomChebPoly(n::Int, ::Type{T}=Float64) where T<:Real = ChebPoly(randn(T, n))

"""
    RandomChebPoly(n, dimension, [T=Float64])

Random polynomial with n node values, each an array of given dimension.
"""
RandomChebPoly(n::Int, dimension::Tuple, ::Type{T}=Float64) where T<:Real =
    ChebPoly([randn(T, dimension) for _ in 1:n])

"""
    (f::ChebPoly)(t::Real)

Evaluate polynomial at t in [0,1] via the barycentric formula.
"""
function (f::ChebPoly)(t::Real)
    # [BT04, eq. 4.2 and section 7]
    numer = zero(f.vals[1])
    denom = 0.0
    @inbounds for j in 1:f.N
        xdiff = t - f.nodes[j]
        # to avoid division by zero at the nodes, return the node value directly
        xdiff == 0 && return f.vals[j]
        temp = f.bary[j] / xdiff
        numer = numer .+ temp .* f.vals[j]
        denom += temp
    end
    return numer ./ denom
end

# Lagrange basis
# [BT04, eq. 4.2]
function basis_weights(f::ChebPoly, t::Real)
    w = zeros(Float64, f.N)
    @inbounds for j in 1:f.N
        xdiff = t - f.nodes[j]
        if xdiff == 0
            fill!(w, 0.0)
            w[j] = 1.0
            return w
        end
        w[j] = f.bary[j] / xdiff
    end
    w ./= sum(w)
    return w
end

nodes(f::ChebPoly) = f.nodes

reconstruct(f::ChebPoly{T}, new_vals) where T =
    ChebPoly{T}(new_vals, f.N, f.nodes, f.bary, f.basis_integrals)

random_like(::Type{ChebPoly}, n::Int, ::Type{T}=Float64) where T<:Real = RandomChebPoly(n, T)
random_like(::Type{ChebPoly}, n::Int, dims::Tuple, ::Type{T}=Float64) where T<:Real =
    RandomChebPoly(n, dims, T)

LinearAlgebra.norm(f::ChebPoly) = sqrt(sum(f.basis_integrals .* _norm_integrand(f)))

function LinearAlgebra.dot(f::ChebPoly, g::ChebPoly)
    length(f) != length(g) && throw(DimensionMismatch("ChebPoly lengths must match"))
    return sum(f.basis_integrals .* _dot_integrand(f, g))
end

end
