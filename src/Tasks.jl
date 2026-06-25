"""
Define training tasks. Each pairs a data generator with its input/output dimensions,
under a uniform interface so experiments run on all tasks. See `AbstractTask` for what
a new task must implement.
"""
module Tasks

using LinearAlgebra
using Random
using ..AbstractRepresentation
using ..SplineRepresentation
using ..ActivationFunctions
using ..ODEModel

export AbstractTask
export DeterminantTask, NestedDeterminantTask
export CofactorDeterminantTask, RandomMinorDeterminantTask
export generate, input_dim, output_dim, task_name
export compute_target, has_target_fn

"""
    AbstractTask

Supertype for training tasks. For a new task, define:
- `generate(task, k; rng) -> (inputs, targets)`
- `input_dim(task) -> Int`
- `output_dim(task) -> Int`
- `task_name(task) -> String`
- `compute_target(task, input) -> AbstractVector`

Targets are `AbstractVector{Float64}` of length `output_dim(task)`.
We use Float64 all the time to keep the uniform interface.
"""
abstract type AbstractTask end

"""
    has_target_fn(task) → Bool

Whether `compute_target` is supported. False for tasks whose targets exist only as data.
"""
has_target_fn(::AbstractTask) = true

"""
    DeterminantTask(n; input_scale=1.0)

Input: n×n random matrix. Target: [det(M)].
"""
struct DeterminantTask <: AbstractTask
    n::Int
    input_scale::Float64
end
DeterminantTask(n::Int; input_scale::Float64=1.0) = DeterminantTask(n, input_scale)

input_dim(t::DeterminantTask) = t.n^2
output_dim(::DeterminantTask) = 1
task_name(t::DeterminantTask) = "determinant_$(t.n)"

compute_target(::DeterminantTask, M::AbstractMatrix) = [det(M)]

function generate(t::DeterminantTask, k::Int; rng::AbstractRNG=Random.default_rng())
    matrices = [t.input_scale .* randn(rng, t.n, t.n) for _ in 1:k]
    targets = [compute_target(t, M) for M in matrices]
    return matrices, targets
end

"""
    NestedDeterminantTask(n; levels=n, input_scale=1.0)

Input: n×n random matrix. Target: determinants of the top left minors of each dimension 1-n
Each is divided by √(k!) (k = block size) to control variance.
"""
struct NestedDeterminantTask <: AbstractTask
    n::Int
    levels::Int
    input_scale::Float64
end
function NestedDeterminantTask(n::Int; levels::Int=n, input_scale::Float64=1.0)
    1 <= levels <= n || error("NestedDeterminantTask: levels=$levels out of range 1:$n")
    return NestedDeterminantTask(n, levels, input_scale)
end

input_dim(t::NestedDeterminantTask) = t.n^2
output_dim(t::NestedDeterminantTask) = t.levels
task_name(t::NestedDeterminantTask) =
    t.levels == t.n ? "nested_determinant_$(t.n)" : "nested_determinant_$(t.n)_L$(t.levels)"

compute_target(t::NestedDeterminantTask, M::AbstractMatrix) =
    [det(M[k:t.n, k:t.n]) / sqrt(factorial(t.n - k + 1)) for k in 1:t.levels]

function generate(t::NestedDeterminantTask, k::Int; rng::AbstractRNG=Random.default_rng())
    matrices = [t.input_scale .* randn(rng, t.n, t.n) for _ in 1:k]
    targets = [compute_target(t, M) for M in matrices]
    return matrices, targets
end

"""
    CofactorDeterminantTask(n; levels=n, input_scale=1.0)

Input: n×n random matrix. Target: det(M) followed by the first-row cofactors
Each is divided by √(k!) (k = block size) to control variance.
"""
struct CofactorDeterminantTask <: AbstractTask
    n::Int
    levels::Int
    input_scale::Float64
end
function CofactorDeterminantTask(n::Int; levels::Int=n, input_scale::Float64=1.0)
    1 <= levels <= n || error("CofactorDeterminantTask: levels=$levels out of range 1:$n")
    return CofactorDeterminantTask(n, levels, input_scale)
end

input_dim(t::CofactorDeterminantTask) = t.n^2
output_dim(t::CofactorDeterminantTask) = 1 + sum(k for k in t.n:-1:(t.n-t.levels+2); init=0)
task_name(t::CofactorDeterminantTask) =
    t.levels == t.n ? "cofactor_determinant_$(t.n)" : "cofactor_determinant_$(t.n)_L$(t.levels)"

function compute_target(t::CofactorDeterminantTask, M::AbstractMatrix)
    n = t.n
    out = [det(M) / sqrt(factorial(n))]
    for k in n:-1:(n-t.levels+2)
        B = M[n-k+1:n, n-k+1:n]
        for j in 1:k
            minor = B[2:k, setdiff(1:k, j)]
            push!(out, (-1)^(1 + j) * det(minor) / sqrt(factorial(k - 1)))
        end
    end
    return out
end

function generate(t::CofactorDeterminantTask, k::Int; rng::AbstractRNG=Random.default_rng())
    matrices = [t.input_scale .* randn(rng, t.n, t.n) for _ in 1:k]
    targets = [compute_target(t, M) for M in matrices]
    return matrices, targets
end

"""
    RandomMinorDeterminantTask(n; m=2, input_scale=1.0, seed=0)

Input: n×n random matrix. Target: NestedDeterminantTask plus m random k×k minors per level.
Each is divided by √(k!) (k = block size) to control variance.
"""
struct RandomMinorDeterminantTask <: AbstractTask
    n::Int
    m::Int
    input_scale::Float64
    rows::Vector{Vector{Int}}
    cols::Vector{Vector{Int}}
end

function RandomMinorDeterminantTask(n::Int; m::Int=2, input_scale::Float64=1.0, seed::Int=0)
    rng = Xoshiro(seed)
    rows = Vector{Int}[]
    cols = Vector{Int}[]
    for k in 2:n-1
        lo = n - k
        for _ in 1:m
            push!(rows, sort(lo .+ randperm(rng, k + 1)[1:k] .- 1))
            push!(cols, sort(lo .+ randperm(rng, k + 1)[1:k] .- 1))
        end
    end
    return RandomMinorDeterminantTask(n, m, input_scale, rows, cols)
end

input_dim(t::RandomMinorDeterminantTask) = t.n^2
output_dim(t::RandomMinorDeterminantTask) = t.n + length(t.rows)
task_name(t::RandomMinorDeterminantTask) = "random_minor_determinant_$(t.n)"

function compute_target(t::RandomMinorDeterminantTask, M::AbstractMatrix)
    n = t.n
    out = [det(M[k:n, k:n]) / sqrt(factorial(n - k + 1)) for k in 1:n]
    for (rs, cs) in zip(t.rows, t.cols)
        push!(out, det(M[rs, cs]) / sqrt(factorial(length(rs))))
    end
    return out
end

function generate(t::RandomMinorDeterminantTask, k::Int; rng::AbstractRNG=Random.default_rng())
    matrices = [t.input_scale .* randn(rng, t.n, t.n) for _ in 1:k]
    targets = [compute_target(t, M) for M in matrices]
    return matrices, targets
end

end
