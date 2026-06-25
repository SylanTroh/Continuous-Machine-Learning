"""
A standard discrete neural network.
Used as a baseline against the continuous model.
Should be identical to the continuous model using Euler.
"""
module DiscreteNN

using LinearAlgebra
using Random
using ..ActivationFunctions
using ..GradientUtils
import ..GradientUtils: Adam, apply_gradients!
using ..GradientUtils: adam_update!
import ..TrainingUtils: compute_error
import ..ModelUtils: n_params

export DiscreteNetwork, forward, backprop, train_discrete

"""
    DiscreteNetwork

K-layer residual network: x_[k+1] = x_k + h·σ(A_k·x_k + B_k), h = 1/K.
x_1 = W_in·a, y = W_out·x_[K+1].
"""
mutable struct DiscreteNetwork{T_ACT}
    As::Vector{Matrix{Float64}}
    Bs::Vector{Vector{Float64}}
    W_in::Matrix{Float64}
    W_out::Matrix{Float64}
    sigma::T_ACT
end

"""
    DiscreteNetwork(input_dim, hidden_dim, K, output_dim; sigma=relu, init_scale=0.01, rng=...)

Construct a K-layer network. Parameters initialized with `init_scale`.
"""
function DiscreteNetwork(input_dim::Int, hidden_dim::Int, K::Int, output_dim::Int;
    sigma=relu, init_scale::Float64=0.01,
    rng::AbstractRNG=Random.default_rng())
    As = [init_scale .* randn(rng, hidden_dim, hidden_dim) for _ in 1:K]
    Bs = [init_scale .* randn(rng, hidden_dim) for _ in 1:K]
    W_in = init_scale .* randn(rng, hidden_dim, input_dim)
    W_out = init_scale .* randn(rng, output_dim, hidden_dim)
    return DiscreteNetwork(As, Bs, W_in, W_out, sigma)
end

"""
    forward(model, input) → (xs::Vector{Vector}, pre::Vector{Vector}, output::Vector)

Evaluate the neural network.
"""
function forward(model::DiscreteNetwork, input)
    K = length(model.As)
    h = 1.0 / K
    a = model.W_in * vec(input)
    xs = Vector{Vector{Float64}}(undef, K + 1)
    pre = Vector{Vector{Float64}}(undef, K)
    xs[1] = a
    for k in 1:K
        pre[k] = model.As[k] * xs[k] .+ model.Bs[k]
        xs[k+1] = xs[k] .+ h .* model.sigma.(pre[k])
    end
    output = model.W_out * xs[K+1]
    return xs, pre, output
end

"""
    backprop(model, inputs, targets)
        → (grad_As, grad_Bs, grad_W_in, grad_W_out)

Compute squared-error gradient.
"""
function backprop(model::DiscreteNetwork,
    inputs::AbstractVector, targets::AbstractVector)
    K = length(model.As)
    grad_As = [zero(A) for A in model.As]
    grad_Bs = [zero(B) for B in model.Bs]
    grad_W_in = zero(model.W_in)
    grad_W_out = zero(model.W_out)

    h = 1.0 / K
    for n in eachindex(inputs)
        xs, pre, output = forward(model, inputs[n])

        r = output .- targets[n]
        grad_W_out .+= 2.0 .* r * xs[K+1]'
        costate = 2.0 .* (model.W_out' * r)

        # Gradient per layer; costate carries dE/dx_[k+1], the skip adds the identity path
        for k in K:-1:1
            s = activation_derivative(model.sigma, pre[k]) .* costate
            grad_As[k] .+= h .* (s * xs[k]')
            grad_Bs[k] .+= h .* s
            costate = costate .+ h .* (model.As[k]' * s)
        end

        grad_W_in .+= costate * vec(inputs[n])'
    end

    return grad_As, grad_Bs, grad_W_in, grad_W_out
end

function Adam(model::DiscreteNetwork; β₁::Float64=0.9, β₂::Float64=0.999,
    epsilon::Float64=1e-8)
    Adam(β₁, β₂, epsilon, 0,
        [zero(A) for A in model.As],
        [zero(A) for A in model.As],
        [zero(B) for B in model.Bs],
        [zero(B) for B in model.Bs],
        zero(model.W_in), zero(model.W_in),
        zero(model.W_out), zero(model.W_out))
end

function apply_gradients!(opt::Adam, model::DiscreteNetwork, lr,
    grad_As, grad_Bs, grad_W_in, grad_W_out)
    check_gradients([grad_As..., grad_Bs..., grad_W_in, grad_W_out], 100000.0)
    opt.t += 1
    for k in eachindex(grad_As)
        adam_update!(opt, lr, model.As[k], opt.m_A[k], opt.v_A[k], grad_As[k])
    end
    for k in eachindex(grad_Bs)
        adam_update!(opt, lr, model.Bs[k], opt.m_B[k], opt.v_B[k], grad_Bs[k])
    end
    adam_update!(opt, lr, model.W_in, opt.m_W_in, opt.v_W_in, grad_W_in)
    adam_update!(opt, lr, model.W_out, opt.m_W_out, opt.v_W_out, grad_W_out)
end

function compute_error(model::DiscreteNetwork, inputs, targets)
    return batch_sum(inputs) do n
        _, _, output = forward(model, inputs[n])
        sum(abs2, output .- targets[n])
    end
end

function n_params(model::DiscreteNetwork)
    nA = sum(length, model.As)
    nB = sum(length, model.Bs)
    return nA + nB + length(model.W_in) + length(model.W_out)
end

"""
    train_discrete(model, generate_batch; kwargs...) → (model, errors)
"""
function train_discrete(model::DiscreteNetwork, generate_batch;
    learning_rate=1e-3, n_epochs=100, verbose=10,
    num_samples=128, patience=250, lr_decay=0.75, min_lr=1e-8)
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)

    opt = Adam(model)
    eval_inputs, eval_targets = generate_batch()

    function step!(lr)
        inputs, targets = generate_batch()
        grad_As, grad_Bs, grad_Wi, grad_Wo = backprop(model, inputs, targets)
        apply_gradients!(opt, model, lr, grad_As, grad_Bs, grad_Wi, grad_Wo)
        return compute_error(model, eval_inputs, eval_targets)
    end

    errors = train_loop(step!, n_epochs, num_samples;
        learning_rate, verbose, patience, lr_decay, min_lr, label="Discrete")
    return model, errors
end

end
