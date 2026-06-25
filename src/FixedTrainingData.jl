"""
Generate a fixed set of data so that training errors are comparable.
Also used to debug overfitting/memorization.
"""
module FixedTrainingData

using Random
using ..Tasks: AbstractTask, generate

export DataPool, sample_from_pool, generate_sampler, build_data_pool

"""
    DataPool{TI, TT}

Stores training data.
"""
struct DataPool{TI,TT}
    inputs::Vector{TI}
    targets::Vector{TT}
end

Base.length(p::DataPool) = length(p.inputs)

"""
    build_data_pool(task, n; rng=Random.default_rng()) → DataPool

Generate a data pool with `n` (input, target) pairs.
"""
function build_data_pool(task::AbstractTask, n::Int;
    rng::AbstractRNG=Random.default_rng())
    inputs, targets = generate(task, n; rng=rng)
    return DataPool(inputs, targets)
end

"""
    sample_from_pool(pool, k; rng=Random.default_rng(), replace=true) → (inputs, targets)

Sample k items, with replacement by default.
"""
function sample_from_pool(pool::DataPool, k::Int;
    rng::AbstractRNG=Random.default_rng(), replace::Bool=true)
    n = length(pool)
    idxs = replace ? rand(rng, 1:n, k) : Random.randperm(rng, n)[1:min(k, n)]
    return pool.inputs[idxs], pool.targets[idxs]
end

"""
    generate_sampler(pool, k; rng=Random.default_rng(), replace=true) → Function

Zero-argument sampler for `train_adjoint`.
"""
function generate_sampler(pool::DataPool, k::Int;
    rng::AbstractRNG=Random.default_rng(), replace::Bool=true)
    return () -> sample_from_pool(pool, k; rng=rng, replace=replace)
end

end
