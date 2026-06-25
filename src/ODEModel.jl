"""
The neural network as an ODE: dx/dt = σ(A(t)·x + B(t)), x(0) = a.
A(t) and B(t) are the parameters and can use any AbstractFunctionRepresentation.
W_in and W_out are used to reshape the input and output data.
"""
module ODEModel

using LinearAlgebra
using ..AbstractModels: AbstractModel
using ..AbstractRepresentation
import ..AbstractRepresentation: change_representation
using ..ActivationFunctions

export AbstractFlowModel, NeuralFlowODE, f
export embed_input, readout, state_dim, field_eltype, n_control_points

"""
    AbstractFlowModel

Supertype for every continuous model architecture. A model is defined by a function
`f(model, x, t)` plus the interface below:

- `embed_input(model, Ain)` initial state `x(0)` from input(s) `Ain`
- `readout(model, Xend)` prediction from the final state
- `state_dim(model)` dimension of the integrated state
- `field_eltype(model)` element type of the field (for solver buffers)
- `n_control_points(model)` number of control points (the depth-in-t axis)
"""
abstract type AbstractFlowModel <: AbstractModel end

"""
    NeuralFlowODE

ODE model: dx/dt = σ(A(t)·x + B(t)), x(0) = a.

# Fields
- `A::AbstractFunctionRepresentation{<:Matrix}` matrix-valued component of P(t)
- `B::AbstractFunctionRepresentation{<:Vector}` vector-valued component of P(t)
- `W_in::Matrix`      input embedding  (hidden_dim x input_dim)
- `W_out::Matrix`     output embedding (output_dim x hidden_dim)
- `sigma::Function`   activation function σ
"""
mutable struct NeuralFlowODE{TA<:AbstractFunctionRepresentation,TB<:AbstractFunctionRepresentation,
    TWi<:AbstractMatrix,TWo<:AbstractMatrix,T_ACT} <: AbstractFlowModel
    A::TA
    B::TB
    W_in::TWi
    W_out::TWo
    sigma::T_ACT
end

"""
    NeuralFlowODE(A, B; sigma=relu)

When input, hidden, and output dims are equal.
"""
function NeuralFlowODE(A::AbstractFunctionRepresentation, B::AbstractFunctionRepresentation; sigma=relu)
    d = length(B[1])
    T = eltype(B[1])
    return NeuralFlowODE(A, B, Matrix{T}(I, d, d), Matrix{T}(I, d, d), sigma)
end

NeuralFlowODE(A::AbstractFunctionRepresentation, B::AbstractFunctionRepresentation,
    W_in::AbstractMatrix, W_out::AbstractMatrix; sigma=relu) =
    NeuralFlowODE(A, B, W_in, W_out, sigma)

"""
    f(model, x, t)

Evaluate f(x, t; P) = σ(A(t)·x + B(t))
"""
f(model::NeuralFlowODE, x::AbstractVecOrMat{<:Real}, t::Real) =
    model.sigma.(model.A(t) * x .+ model.B(t))

"""
    change_representation(model, F, n) → NeuralFlowODE

Re-represent A(t), B(t) in basis `F` with `n` coefficients; embeddings and σ are kept.
"""
change_representation(model::NeuralFlowODE, ::Type{F}, n::Int) where {F<:AbstractFunctionRepresentation} =
    NeuralFlowODE(change_representation(model.A, F, n),
        change_representation(model.B, F, n),
        model.W_in, model.W_out, model.sigma)

"""
    embed_input(model, Ain) → x(0)

Lift input(s) `Ain` into the initial state: `x(0) = W_in·a`.
"""
embed_input(model::NeuralFlowODE, Ain) = model.W_in * Ain

"""
    readout(model, Xend) → prediction

Map the final state to the prediction. NeuralFlowODE: `W_out·x(1)`.
"""
readout(model::NeuralFlowODE, Xend) = model.W_out * Xend

"""
    state_dim(model) → Int

Dimension of the integrated state. NeuralFlowODE: the hidden dimension.
"""
state_dim(model::NeuralFlowODE) = length(model.B[1])

"""
    field_eltype(model) → Type

Element type of the vector field, for allocating solver buffers.
"""
field_eltype(model::NeuralFlowODE) = eltype(model.B[1])

"""
    n_control_points(model) → Int

Number of control points of the time-varying parameters.
"""
n_control_points(model::NeuralFlowODE) = length(model.B)

end
