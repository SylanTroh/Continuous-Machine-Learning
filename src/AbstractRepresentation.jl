"""
Supertype for different function representations.
Parameterizes a function f: [0,1] -> T where T may be Real, Vector, or Matrix.
In order for a function representation to work with the rest of the code,
the following implementations are required:
- `(f::F)(t::Real)` - evaluate f at t
- `basis_weights(f, t)` - basis functions b_k(t) such that f(t) = Σ b_k(t)·vals(f)[k].
- `nodes(f)` - the points t_k where each coefficient k acts.
- `reconstruct(f, new_vals)` - rebuild the same type around new coefficients
  for most types, just do: reconstruct(f::F, new_vals) = F(new_vals)
- `random_like(::Type{F}, n, dims, T)` - create a random function with n coefficients of size dims
- `LinearAlgebra.norm`, `LinearAlgebra.dot` - L² norm and inner product
"""
module AbstractRepresentation

using LinearAlgebra
using ChainRulesCore: @non_differentiable

export AbstractFunctionRepresentation
export vals, basis_weights, nodes, reconstruct, random_like, change_representation
export device_weights, device_basis_weights, BasisCache

abstract type AbstractFunctionRepresentation{T} <: AbstractVector{T} end

vals(f::AbstractFunctionRepresentation) = f.vals

function basis_weights end
function nodes end
function reconstruct end
function random_like end

"""
    change_representation(f, F, n) → F

Reconstruct f under representation F.
"""
function change_representation(f::AbstractFunctionRepresentation, ::Type{F},
    n::Int) where {F<:AbstractFunctionRepresentation}
    proto = _prototype(F, n, vals(f)[1])
    return reconstruct(proto, [f(t) for t in nodes(proto)])
end

_prototype(::Type{F}, n::Int, v::Real) where {F} = random_like(F, n, typeof(float(v)))
_prototype(::Type{F}, n::Int, v::AbstractArray) where {F} = random_like(F, n, size(v), eltype(v))

#The basis depends only on t, never on the coefficients, so AD can treat it as constant.
@non_differentiable basis_weights(::AbstractFunctionRepresentation, ::Real)

#Keeps things hardware agnostic for the GPU code
device_weights(w::AbstractVector, x::AbstractArray) =
    copyto!(similar(x, eltype(x), length(w)), w)
@non_differentiable device_weights(::Any, ::Any)
device_basis_weights(f::AbstractFunctionRepresentation, t::Real, x::AbstractArray) =
    device_weights(basis_weights(f, t), x)
@non_differentiable device_basis_weights(::Any, ::Any, ::Any)

"""
    BasisCache(rep, x)

Caches `device_basis_weights(rep, t, x)` by t.
"""
struct BasisCache{F<:AbstractFunctionRepresentation,W<:AbstractVector}
    rep::F
    table::Dict{Float64,W}
end

BasisCache(rep::AbstractFunctionRepresentation, x::AbstractArray) =
    BasisCache(rep, Dict{Float64,typeof(device_basis_weights(rep, 0.0, x))}())

function device_basis_weights(c::BasisCache, t::Real, x::AbstractArray)
    w = get(c.table, Float64(t), nothing)
    w === nothing || return w
    return c.table[Float64(t)] = device_basis_weights(c.rep, t, x)
end

_norm_integrand(f::AbstractFunctionRepresentation{<:Real}) = abs2.(vals(f))
_norm_integrand(f::AbstractFunctionRepresentation{<:AbstractArray}) = [LinearAlgebra.norm(v)^2 for v in vals(f)]
_dot_integrand(f::AbstractFunctionRepresentation{<:Real}, g::AbstractFunctionRepresentation{<:Real}) = vals(f) .* vals(g)
_dot_integrand(f::AbstractFunctionRepresentation{<:AbstractArray}, g::AbstractFunctionRepresentation{<:AbstractArray}) = [LinearAlgebra.dot(u, v) for (u, v) in zip(vals(f), vals(g))]

Base.getindex(f::AbstractFunctionRepresentation, idx) = getindex(vals(f), idx)
Base.setindex!(f::AbstractFunctionRepresentation, v, idx) = setindex!(vals(f), v, idx)
Base.size(f::AbstractFunctionRepresentation) = size(vals(f))
Base.iterate(f::AbstractFunctionRepresentation) = iterate(vals(f))
Base.iterate(f::AbstractFunctionRepresentation, state) = iterate(vals(f), state)

function Base.:+(f::F, g::F) where {F<:AbstractFunctionRepresentation}
    length(f) != length(g) && throw(DimensionMismatch("$(nameof(F)) lengths must match"))
    return reconstruct(f, vals(f) .+ vals(g))
end

function Base.:-(f::F, g::F) where {F<:AbstractFunctionRepresentation}
    length(f) != length(g) && throw(DimensionMismatch("$(nameof(F)) lengths must match"))
    return reconstruct(f, vals(f) .- vals(g))
end

Base.:-(f::AbstractFunctionRepresentation) = reconstruct(f, -vals(f))
Base.:*(c::Number, f::AbstractFunctionRepresentation) = reconstruct(f, c .* vals(f))
Base.:*(f::AbstractFunctionRepresentation, c::Number) = c * f

function Base.:/(f::AbstractFunctionRepresentation, c::Number)
    c == 0 && throw(DivideError())
    return (1 / c) * f
end

function Base.show(io::IO, f::AbstractFunctionRepresentation{T}) where T
    print(io, "$(nameof(typeof(f))){$T}($(length(f)))")
end

function Base.show(io::IO, ::MIME"text/plain", f::AbstractFunctionRepresentation{T}) where T
    println(io, "$(nameof(typeof(f))){$T}, $(length(f)) values on [0, 1]")
    if length(f) <= 10
        println(io, "  Values: $(vals(f))")
    else
        println(io, "  Values: $(vals(f)[1:3]) ... $(vals(f)[end-2:end])")
    end
    println(io, "  L2 norm: $(round(LinearAlgebra.norm(f), digits=6))")
end

end
