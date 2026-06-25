"""
    ResBlockModel

The depth N residual block model:

    dx/dt = F(x,t),   F(x,t) = σ(A_N(t)·σ(… σ(A_1(t)·x + B_1(t)) …) + B_N(t))

"""
module ResBlockModel

using LinearAlgebra
using ..AbstractRepresentation
using ..ActivationFunctions
using ..ODEModel: AbstractFlowModel
using ..BlockField
import ..ODEModel: f, embed_input, readout, state_dim, field_eltype, n_control_points

export ResBlockFlow

"""
    ResBlockFlow

# Fields
- `spec::BlockSpec` block depth/widths + flat-coefficient layout
- `As::Vector`, `Bs::Vector` per-layer weight/bias representations
- `W_in`, `W_out` input/output embeddings
- `sigma` activation function
"""
mutable struct ResBlockFlow{RA<:AbstractFunctionRepresentation,RB<:AbstractFunctionRepresentation,
    TWi<:AbstractMatrix,TWo<:AbstractMatrix,T_ACT} <: AbstractFlowModel
    spec::BlockSpec
    As::Vector{RA}
    Bs::Vector{RB}
    W_in::TWi
    W_out::TWo
    sigma::T_ACT
end

"""
    f(model::ResBlockFlow, x, t)
"""
function f(model::ResBlockFlow, x::AbstractVecOrMat{<:Real}, t::Real)
    Y = x
    @inbounds for l in eachindex(model.As)
        Y = model.sigma.(model.As[l](t) * Y .+ model.Bs[l](t))
    end
    return Y
end

embed_input(model::ResBlockFlow, Ain) = model.W_in * Ain
readout(model::ResBlockFlow, Xend) = model.W_out * Xend
state_dim(model::ResBlockFlow) = model.spec.d
field_eltype(model::ResBlockFlow) = eltype(model.Bs[1][1])
n_control_points(model::ResBlockFlow) = model.spec.ncp

end
