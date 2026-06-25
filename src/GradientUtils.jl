"""
Utilities for gradient descent. SGD and the Adam optimizer.
"""
module GradientUtils

using LinearAlgebra
using ComponentArrays
using Base.ScopedValues: ScopedValue, with
using ..AbstractRepresentation
using ..ODEModel

export check_gradients, apply_gradients!, apply_gradients_perseed!, train_loop, Adam, AdamFlat, SGD
export batch_sum
export CurveSink, with_curve_sink, take_curves, record_curve!

# Thread-safe sink so every `train_loop` records its loss curve.
struct CurveSink
    lock::ReentrantLock
    curves::Vector{NamedTuple}
end
CurveSink() = CurveSink(ReentrantLock(), NamedTuple[])

const ACTIVE_CURVE_SINK = ScopedValue{Union{Nothing,CurveSink}}(nothing)

"""
    record_curve!(curve; tag=nothing, kind=:loss) → curve

Append `curve` to the active `CurveSink`, or no-op if none is active.
"""
function record_curve!(curve; tag=nothing, kind::Symbol=:loss)
    sink = ACTIVE_CURVE_SINK[]
    (sink === nothing || isempty(curve)) && return curve
    lock(sink.lock) do
        push!(sink.curves, (; tag, kind, curve=Vector{Float64}(curve)))
    end
    return curve
end

"""
    with_curve_sink(f, sink::CurveSink)

Run `f` with `sink` installed as the active curve sink.
"""
with_curve_sink(f, sink::CurveSink) = with(f, ACTIVE_CURVE_SINK => sink)

"""
    take_curves(sink) → Vector{NamedTuple}
"""
take_curves(sink::CurveSink) = sink.curves

"""
    batch_sum(f, inputs) → Float64

Threaded sum of `f(n)`.
"""
function batch_sum(f, inputs)
    nthreads = Threads.nthreads()
    chunk_size = max(1, length(inputs) ÷ nthreads)
    chunks = [(i:min(i + chunk_size - 1, length(inputs))) for i in 1:chunk_size:length(inputs)]
    partial = zeros(length(chunks))
    #Multiple OpenBLAS threads inside a multithreaded region can cause a crash.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for c in eachindex(chunks)
            partial[c] = sum(f(n) for n in chunks[c])
        end
    finally
        BLAS.set_num_threads(blas_threads)
    end
    return sum(partial)
end

"""
    check_gradients(grads, max_norm::Float64)

Warn if any gradient block's norm exceeds `max_norm`.
"""
function check_gradients(grads, max_norm::Float64)
    for g in grads
        _check_block(g, max_norm)
    end
end

function _check_block(g, max_norm)
    n = norm(g)
    n > max_norm && @warn "Gradient norm $(round(n, digits=4)) exceeds $max_norm"
    return nothing
end

# Flat version; unwrap so norm doesn't scalar-iterate on device arrays.
check_gradients(g::ComponentVector, max_norm::Float64) = _check_block(getdata(g), max_norm)

"""
    Adam(model; β₁=0.9, β₂=0.999, epsilon=1e-8)

Adam state for a model.
"""
mutable struct Adam{TA,TB,TWi<:AbstractMatrix,TWo<:AbstractMatrix}
    β₁::Float64
    β₂::Float64
    epsilon::Float64
    t::Int
    m_A::Vector{TA}
    v_A::Vector{TA}
    m_B::Vector{TB}
    v_B::Vector{TB}
    m_W_in::TWi
    v_W_in::TWi
    m_W_out::TWo
    v_W_out::TWo
end

function Adam(model::NeuralFlowODE; β₁::Float64=0.9, β₂::Float64=0.999,
    epsilon::Float64=1e-8)
    Adam(β₁, β₂, epsilon, 0,
        [zero(Ak) for Ak in vals(model.A)],
        [zero(Ak) for Ak in vals(model.A)],
        [zero(Bk) for Bk in vals(model.B)],
        [zero(Bk) for Bk in vals(model.B)],
        zero(model.W_in), zero(model.W_in),
        zero(model.W_out), zero(model.W_out))
end

@inline function adam_update!(opt::Adam, lr, P, m, v, g)
    m .= opt.β₁ .* m .+ (1.0 - opt.β₁) .* g
    v .= opt.β₂ .* v .+ (1.0 - opt.β₂) .* g .^ 2
    P .-= lr .* (m ./ (1.0 - opt.β₁^opt.t)) ./ (sqrt.(v ./ (1.0 - opt.β₂^opt.t)) .+ opt.epsilon)
end

function apply_gradients!(opt::Adam, model::NeuralFlowODE, lr,
    grad_A, grad_B, grad_W_in, grad_W_out)
    check_gradients([grad_A..., grad_B..., grad_W_in, grad_W_out], 100000.0)
    opt.t += 1
    for k in eachindex(grad_A)
        adam_update!(opt, lr, vals(model.A)[k], opt.m_A[k], opt.v_A[k], grad_A[k])
    end
    for k in eachindex(grad_B)
        adam_update!(opt, lr, vals(model.B)[k], opt.m_B[k], opt.v_B[k], grad_B[k])
    end
    adam_update!(opt, lr, model.W_in, opt.m_W_in, opt.v_W_in, grad_W_in)
    adam_update!(opt, lr, model.W_out, opt.m_W_out, opt.v_W_out, grad_W_out)
end

"""
    AdamFlat(p; β₁=0.9, β₂=0.999, epsilon=1e-8)

Adam state over a flat parameter vector.
"""
mutable struct AdamFlat{TV<:AbstractVector}
    β₁::Float64
    β₂::Float64
    epsilon::Float64
    t::Int
    m::TV
    v::TV
end

AdamFlat(p::AbstractVector; β₁::Float64=0.9, β₂::Float64=0.999,
    epsilon::Float64=1e-8) =
    AdamFlat(β₁, β₂, epsilon, 0, zero(p), zero(p))

function apply_gradients!(opt::AdamFlat, p::AbstractVector, lr, g::ComponentVector)
    check_gradients(g, 100000.0)
    opt.t += 1
    opt.m .= opt.β₁ .* opt.m .+ (1.0 - opt.β₁) .* g
    opt.v .= opt.β₂ .* opt.v .+ (1.0 - opt.β₂) .* g .^ 2
    p .-= lr .* (opt.m ./ (1.0 - opt.β₁^opt.t)) ./
          (sqrt.(opt.v ./ (1.0 - opt.β₂^opt.t)) .+ opt.epsilon)
end

# Per-seed Adam step
function _perseed_step!(p, m, v, lr, mc, vc, eps)
    K = size(p)[end]
    lrr = reshape(lr, ntuple(_ -> 1, ndims(p) - 1)..., K)
    @. p -= lrr * (m / mc) / (sqrt(v / vc) + eps)
    return p
end

function apply_gradients_perseed!(opt::AdamFlat, P::ComponentVector, lr, g::ComponentVector;
    train_win::Bool=false)
    check_gradients(g, 100000.0)
    opt.t += 1
    opt.m .= opt.β₁ .* opt.m .+ (1.0 - opt.β₁) .* g
    opt.v .= opt.β₂ .* opt.v .+ (1.0 - opt.β₂) .* g .^ 2
    mc = 1.0 - opt.β₁^opt.t
    vc = 1.0 - opt.β₂^opt.t
    _perseed_step!(P.A, opt.m.A, opt.v.A, lr, mc, vc, opt.epsilon)
    _perseed_step!(P.B, opt.m.B, opt.v.B, lr, mc, vc, opt.epsilon)
    _perseed_step!(P.W_out, opt.m.W_out, opt.v.W_out, lr, mc, vc, opt.epsilon)
    train_win && _perseed_step!(P.W_in, opt.m.W_in, opt.v.W_in, lr, mc, vc, opt.epsilon)
    return P
end

"""
    SGD(model)

Stochastic gradient descent. Stateless.
"""
struct SGD end
SGD(::NeuralFlowODE) = SGD()
SGD(::AbstractVector) = SGD()

function apply_gradients!(::SGD, p::AbstractVector, lr, g::ComponentVector)
    check_gradients(g, 100000.0)
    p .-= lr .* g
end

function apply_gradients!(::SGD, model::NeuralFlowODE, lr,
    grad_A, grad_B, grad_W_in, grad_W_out)
    check_gradients([grad_A..., grad_B..., grad_W_in, grad_W_out], 100000.0)
    for k in eachindex(grad_A)
        vals(model.A)[k] .-= lr .* grad_A[k]
    end
    for k in eachindex(grad_B)
        vals(model.B)[k] .-= lr .* grad_B[k]
    end
    model.W_in .-= lr .* grad_W_in
    model.W_out .-= lr .* grad_W_out
end

"""
    train_loop(step!, n_epochs, num_samples; kwargs...) → errors

Training loop. step! runs one epoch and returns E(P). `on_epoch(E, epoch, errors)` fires
after each epoch; returning `:stop` ends training early.
"""
function train_loop(step!, n_epochs::Int, num_samples::Int;
    learning_rate=0.01, verbose=10,
    patience=250, lr_decay=0.75, min_lr=1e-8, escape_frac=0.9,
    label="",
    on_epoch::Union{Nothing,Function}=nothing)
    errors = Float64[]
    best = Inf
    stale = 0
    lr = Float64(learning_rate)
    prefix = isempty(label) ? "" : "$label "
    e_first = NaN
    escaped = escape_frac >= 1

    try
        for epoch in 1:n_epochs
            E = step!(lr)
            push!(errors, E)
            isnan(e_first) && (e_first = E)

            if E < best
                best = E
                stale = 0
            else
                stale += 1
            end

            if !escaped
                if best < escape_frac * e_first
                    escaped = true
                else
                    stale = 0
                end
            end

            if escaped && stale >= patience
                new_lr = max(lr * lr_decay, min_lr)
                if new_lr < lr
                    verbose > 0 && @info "$(prefix)Reducing LR $(round(lr, digits=8)) → $(round(new_lr, digits=8))"
                    lr = new_lr
                    stale = 0
                end
            end

            if verbose > 0 && epoch % verbose == 0
                println("$(prefix)Epoch $epoch: Error = $(round(E, digits=6)), " *
                        "Avg = $(round(E/num_samples, digits=6)), LR = $(round(lr, digits=8))")
            end

            if on_epoch !== nothing && on_epoch(E, epoch, errors) === :stop
                break
            end
        end
    catch e
        isa(e, InterruptException) || rethrow()
        @info "$(isempty(label) ? "Training" : label) interrupted."
    end

    record_curve!(errors; tag=label, kind=:loss)
    return errors
end

end
