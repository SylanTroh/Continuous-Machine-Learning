"""
Train on the MNIST dataset as a task
"""
module MNISTData

using Random
using MLDatasets: MNIST
using ..ODEModel
using ..Tasks: AbstractTask
import ..Tasks: generate, input_dim, output_dim, task_name, compute_target, has_target_fn
using ..Adjoint: batched_predict

export MNISTTask, load_mnist, onehot, accuracy

"""
    MNISTTask(images, labels; name="mnist")

`images`: 784 × n matrix, one flattened digit per column, pixels in [0, 1].
`labels`: digits 0-9.
"""
struct MNISTTask <: AbstractTask
    images::Matrix{Float32}
    labels::Vector{Int}
    name::String
end

MNISTTask(images::AbstractMatrix, labels::AbstractVector; name::String="mnist") =
    MNISTTask(Float32.(images), Int.(labels), name)

"""
    load_mnist(; split=:train, n=:all) → MNISTTask

Load MNIST via the MLDatasets library (downloads on first use; set
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" to consent non-interactively).
"""
function load_mnist(; split::Symbol=:train, n::Union{Int,Symbol}=:all)
    data = MNIST(split=split)
    images = reshape(data.features, 28 * 28, :)
    labels = Vector{Int}(data.targets)
    if n !== :all
        images = images[:, 1:n]
        labels = labels[1:n]
    end
    return MNISTTask(Matrix{Float32}(images), labels, "mnist_$(split)")
end

input_dim(::MNISTTask) = 28 * 28
output_dim(::MNISTTask) = 10
task_name(t::MNISTTask) = t.name
has_target_fn(::MNISTTask) = false
compute_target(::MNISTTask, _) =
    error("MNIST labels are data, not a function of the input")

"""
    onehot(label) → Vector{Float32}

One-hot encoding of a digit 0-9 as a vector.
"""
function onehot(label::Integer)
    y = zeros(Float32, 10)
    y[label+1] = 1.0f0
    return y
end

# sample from the stored set with replacement
function generate(t::MNISTTask, k::Int; rng::AbstractRNG=Random.default_rng())
    idx = rand(rng, 1:size(t.images, 2), k)
    return [t.images[:, i] for i in idx], [onehot(t.labels[i]) for i in idx]
end

"""
    accuracy(model, inputs, targets; solver=Tsit5(), reltol, abstol) → Float64

Fraction of samples whose predicted class matches the target.
"""
function accuracy(model::NeuralFlowODE, inputs::Vector, targets::Vector; kwargs...)
    pred = batched_predict(model, inputs; kwargs...)
    Y = reduce(hcat, targets)
    correct = sum(argmax(pred[:, n]) == argmax(Y[:, n]) for n in axes(pred, 2))
    return correct / size(pred, 2)
end

end
