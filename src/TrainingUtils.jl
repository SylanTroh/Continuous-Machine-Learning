"""
Model-training utilities: the error functional `E(P)` and its decomposition over
training / near-training / fresh input sets.
"""
module TrainingUtils

using LinearAlgebra
using Random
using ..ODEModel
using ..Solvers
using ..GradientUtils: batch_sum

export compute_error
export near_training_perturb, decomposed_error

"""
    compute_error(model, inputs, targets; solver=solve_euler)

Squared error E(P) = Σₙ ‖W_out · x(1; aₙ, P) - bₙ‖².
"""
function compute_error(model, inputs, targets; solver=solve_euler)
    time_steps = collect(range(0.0, 1.0, length=4 * n_control_points(model)))
    return batch_sum(inputs) do n
        x0 = embed_input(model, vec(inputs[n]))
        xend = last(solver(model, x0, time_steps)[2])
        sum(abs2, readout(model, xend) .- targets[n])
    end
end

"""
    near_training_perturb(inputs, c; rng=Random.default_rng()) → Vector

Perturb inputs: `[x + c·ξ for x in inputs]`, ξ ~ N(0, 1).
"""
function near_training_perturb(inputs::AbstractVector, c::Real; rng::AbstractRNG=Random.default_rng())
    return [x .+ c .* randn(rng, eltype(x), size(x)) for x in inputs]
end

"""
    decomposed_error(model, train_inputs, train_targets, target_fn, perturb_values,
                     fresh_inputs, fresh_targets; rng) → NamedTuple

Error on three sets, for debugging overfitting/memorization:
- `in_training`: training matrices
- `near_training`: perturbed training matrices
- `fresh`: a random sample
"""
function decomposed_error(model::NeuralFlowODE,
    train_inputs::AbstractVector, train_targets::AbstractVector,
    target_fn,
    perturb_values::AbstractVector{<:Real},
    fresh_inputs::AbstractVector, fresh_targets::AbstractVector;
    rng::AbstractRNG=Random.default_rng())

    in_train = compute_error(model, train_inputs, train_targets) / length(train_inputs)

    near = Pair{Float64,Float64}[]
    for c in perturb_values
        perturbed = near_training_perturb(train_inputs, c; rng=rng)
        new_targets = [target_fn(x) for x in perturbed]
        e = compute_error(model, perturbed, new_targets) / length(perturbed)
        push!(near, Float64(c) => e)
    end

    fresh = compute_error(model, fresh_inputs, fresh_targets) / length(fresh_inputs)

    return (in_training=in_train, near_training=near, fresh=fresh)
end

end
