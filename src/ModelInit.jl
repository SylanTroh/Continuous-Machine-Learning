"""
Model weight initialization schemes.

`init_model` builds a model with randomparameters.
`init_output_layer!` least-squares-fits `W_out`,
`feature_scale_winit!` rescales `W_in` by per-feature scale.
"""
module ModelInit

using LinearAlgebra
using Random
using Statistics
using ..SplineRepresentation
using ..ActivationFunctions
using ..ODEModel
using ..ResBlockModel: ResBlockFlow
using ..BlockField: BlockSpec
using ..Solvers
using ..Tasks: AbstractTask, input_dim, output_dim

export init_model, init_resblock, init_output_layer!, feature_scale_winit!, INIT_SCHEMES

"""
    init_model(task, hidden_dim, ncp; sigma=relu, RepType=Spline,
                     init_scale=0.3, input_gain=1.0, output_gain=1.0, T=Float64) → NeuralFlowODE

Initialize the model with random parameters of element type `T`. `init_scale` sets the
initial gradient scale. Pair with `init_output_layer!` for a data-aware `W_out`.
"""
function init_model(task::AbstractTask, hidden_dim::Int, ncp::Int;
    sigma=relu, RepType::Type=Spline,
    init_scale::Float64=0.3, input_gain::Float64=1.0, output_gain::Float64=1.0,
    T::Type{<:AbstractFloat}=Float64,
    rng::AbstractRNG=Random.default_rng())
    d_in = input_dim(task)
    d_out = output_dim(task)
    s_field = T(init_scale / sqrt(hidden_dim))
    w_in = T(input_gain / sqrt(d_in)) .* randn(rng, T, hidden_dim, d_in)
    w_out = T(output_gain / sqrt(hidden_dim)) .* randn(rng, T, d_out, hidden_dim)
    A_points = [s_field .* randn(rng, T, hidden_dim, hidden_dim) for _ in 1:ncp]
    B_points = [s_field .* randn(rng, T, hidden_dim) for _ in 1:ncp]
    return NeuralFlowODE(
        RepType(A_points), RepType(B_points),
        w_in, w_out,
        sigma)
end

"""
    init_resblock(task, hidden_dim, ncp, mults; sigma=relu, RepType=Spline,
                  init_scale=0.3, input_gain=1.0, output_gain=1.0, T=Float64) → ResBlockFlow

Random depth-N residual-block model. `mults` is the block spec (see `BlockSpec` in BlockField.jl).
Each layer is scaled by its input width.
"""
function init_resblock(task::AbstractTask, hidden_dim::Int, ncp::Int,
    mults::AbstractVector{<:Rational};
    sigma=relu, RepType::Type=Spline,
    init_scale::Float64=0.3, input_gain::Float64=1.0, output_gain::Float64=1.0,
    T::Type{<:AbstractFloat}=Float64, rng::AbstractRNG=Random.default_rng())
    d_in = input_dim(task)
    d_out = output_dim(task)
    d = hidden_dim
    spec = BlockSpec(d, ncp, mults)
    w_in = T(input_gain / sqrt(d_in)) .* randn(rng, T, d, d_in)
    w_out = T(output_gain / sqrt(d)) .* randn(rng, T, d_out, d)
    As = [RepType([T(init_scale / sqrt(m.wi)) .* randn(rng, T, m.wo, m.wi) for _ in 1:ncp]) for m in spec.metas]
    Bs = [RepType([T(init_scale / sqrt(m.wi)) .* randn(rng, T, m.wo) for _ in 1:ncp]) for m in spec.metas]
    return ResBlockFlow(spec, As, Bs, w_in, w_out, sigma)
end

"""
    init_output_layer!(model, inputs, targets; solver=solve_euler, epsilon=1e-6) → model

Init `W_out` by least squares instead of random. `epsilon` on the diagonal avoids
problems when X⋅Xᵀ is singular.
"""
function init_output_layer!(model::NeuralFlowODE, inputs, targets;
    solver=solve_euler, epsilon::Float64=1e-6)
    time_steps = collect(range(0.0, 1.0, length=4 * length(model.B)))
    h = size(model.W_out, 2)
    N = length(inputs)
    d_out = length(targets[1])
    ET = eltype(model.W_out)
    X = Matrix{ET}(undef, h, N)
    Y = Matrix{ET}(undef, d_out, N)
    for n in 1:N
        a = model.W_in * vec(inputs[n])
        X[:, n] = last(solver(model, a, time_steps)[2])
        Y[:, n] = targets[n]
    end
    Wt = (X * X' + ET(epsilon) * I) \ (X * Y')
    model.W_out .= permutedims(Wt)
    return model
end

"""
    feature_scale_winit!(model, X; eps=1e-8) → model

Rescale each column `j` of `model.W_in` by `1/std_j.
"""
function feature_scale_winit!(model, X::AbstractMatrix; eps::Float64=1e-8)
    sd = max.(vec(std(X; dims=2)), eps)
    model.W_in .= model.W_in ./ reshape(sd, 1, :)
    return model
end

# Stack input samples into a d_in × n matrix.
_stack_inputs(inputs) = reduce(hcat, (vec(x) for x in inputs))

"""
    INIT_SCHEMES :: Dict{String,Function}

Named init strategies, each `(model, inputs, targets) -> model` (see `train_run`'s `init_strategy`):

- `"random"`- default
- `"winit"` - feature scale `W_in`
- `"lsq"` - least-squares `W_out`
- `"both"` - `winit` then `lsq`
"""
const INIT_SCHEMES = Dict{String,Function}(
    "random" => (model, inputs, targets) -> model,
    "winit" => (model, inputs, targets) -> feature_scale_winit!(model, _stack_inputs(inputs)),
    "lsq" => (model, inputs, targets) -> init_output_layer!(model, inputs, targets),
    "both" => (model, inputs, targets) ->
        init_output_layer!(feature_scale_winit!(model, _stack_inputs(inputs)), inputs, targets),
)

end
