"""
Batch many independent, same-shape problems into one ODE solve.
"""
module BatchedProblems

using LinearAlgebra
using Adapt
using ComponentArrays
using LinearAlgebra: BLAS
using ChainRulesCore: ChainRulesCore, NoTangent, unthunk
using DifferentialEquations
using SciMLSensitivity
using Zygote
using NNlib: batched_mul, batched_transpose, batched_vec
using ..AbstractRepresentation
using ..ActivationFunctions
using ..ODEModel
using ..BlockField: build_block
using ..ResBlockModel: ResBlockFlow
using ..ModelUtils: param_vector, set_params!
using ..GradientUtils: AdamFlat, apply_gradients!, apply_gradients_perseed!, train_loop, record_curve!
using ..Adjoint: _fixedstep_traj, _fixedstep_adjoint!, loss_value_and_grad,
    squared_error_loss, softmax_crossentropy_loss

export group_by_shape, stack_group, unstack_group!
export batched_group_value_and_gradient, train_batched, batched_heldout_errors,
    batched_heldout_sample_errors
export fast_group_value_and_gradient, fast_group_value
export batched_field_ops, block_stack_group, block_unstack_group!
export block_fast_group_value, block_fast_group_value_and_gradient
export block_fast_group_errors, block_fast_group_predict, train_block_batched
export fastadaptive_group_value_and_gradient, fastadaptive_group_value
export picard_group_value_and_gradient, picard_group_value

# Stack a Vector of N sample vectors into a d×N matrix.
_stack(xs, ::Type{ET}) where {ET} = ET.(reduce(hcat, vec.(xs)))

# Loss per sample: keeps the sample and problem axes (d_out×N×K → N×K)
loss_per_sample(::typeof(squared_error_loss), Z, Y) = dropdims(sum(abs2, Z .- Y; dims=1); dims=1)
function loss_per_sample(::typeof(softmax_crossentropy_loss), Z, Y)
    m = maximum(Z; dims=1)
    lse = m .+ log.(sum(exp.(Z .- m); dims=1))
    return dropdims(lse .- sum(Y .* Z; dims=1); dims=1)
end

_step_opts(::Nothing) = (;)
_step_opts(dt::Real) = (adaptive=false, dt=dt)

# InterpolatingAdjoint stores the forward trajectory and reuses it on the reverse pass.
# This saves time on CPU
# BacksolveAdjoint recomputes the forward trajectory during the backwards pass.
# This saves memory on GPU
_on_device(x) = !(ComponentArrays.getdata(x) isa Array)
_default_sensealg() = InterpolatingAdjoint(autojacvec=ZygoteVJP())
_auto_sensealg(x) = _on_device(x) ?
                    BacksolveAdjoint(autojacvec=ZygoteVJP()) :
                    InterpolatingAdjoint(autojacvec=ZygoteVJP())

# ==== Group key ====
# Two problems batch together when their parameter shapes match (hidden dim, control
# points, representation, activation, output and input dims).
_shape_key(m::NeuralFlowODE) = (length(m.B[1]), length(m.A),
    nameof(typeof(m.A)), m.sigma, size(m.W_out, 1), size(m.W_in, 2))

"""
    group_by_shape(models, gens) → Vector{NamedTuple}

Split into batchable same-shape groups. Each group is `(idx, models, gens)`, where
`idx` are the original positions (so results can be scattered back).
"""
function group_by_shape(models::AbstractVector, gens::AbstractVector)
    length(models) == length(gens) ||
        throw(ArgumentError("models and gens must have equal length"))
    buckets = Dict{Any,Vector{Int}}()
    order = Any[]
    for (i, m) in enumerate(models)
        k = _shape_key(m)
        haskey(buckets, k) || push!(order, k)
        push!(get!(buckets, k, Int[]), i)
    end
    return [(idx=buckets[k], models=models[buckets[k]], gens=gens[buckets[k]])
            for k in order]
end

# ==== Parameter stacking ====
# Same-shape group to one batched ComponentVector P.

"""
    stack_group(models) → P::ComponentVector

Stack a same-shape group: `P.A` is `d×d×nA×K`, `P.B` is `d×nB×K`, `P.W_in` is
`d×d_in×K`, `P.W_out` is `out×d×K`. Inverse of `unstack_group!`.
"""
function stack_group(models::AbstractVector)
    pv = [param_vector(m) for m in models]
    return ComponentVector(
        A=cat((p.A for p in pv)...; dims=4),
        B=cat((p.B for p in pv)...; dims=3),
        W_in=cat((m.W_in for m in models)...; dims=3),
        W_out=cat((m.W_out for m in models)...; dims=3))
end

"""
    unstack_group!(models, P) → models

Write the k-th slice of the batched params back into `models[k]`.
"""
function unstack_group!(models::AbstractVector, P)
    Pc = adapt(Array, P)
    for k in eachindex(models)
        set_params!(models[k], ComponentVector(
            A=Pc.A[:, :, :, k], B=Pc.B[:, :, k],
            W_in=Pc.W_in[:, :, k], W_out=Pc.W_out[:, :, k]))
    end
    return models
end

# ==== Batched eval ====
# Adjoint._f_flat for a batch of K problems.
_batched_At(PA, wA, d, nA) = reshape(batched_vec(reshape(PA, d * d, nA, :), wA), d, d, :)
_batched_Bt(PB, wB) = batched_vec(PB, wB)

function _f_flat_batched(X, P, t, σ, repA, repB, d, nA, nB)
    wA = device_basis_weights(repA, t, X)
    wB = device_basis_weights(repB, t, X)
    At = _batched_At(P.A, wA, d, nA)
    Bt = _batched_Bt(P.B, wB)
    Z = batched_mul(At, X) .+ reshape(Bt, d, 1, :)
    return σ.(Z)
end

# Hand-written per-step VJP, the batched version of the rrule in Adjoint._f_flat.
function ChainRulesCore.rrule(::typeof(_f_flat_batched), X, P, t, σ, repA, repB, d, nA, nB)
    wA = device_basis_weights(repA, t, X)
    wB = device_basis_weights(repB, t, X)
    At = _batched_At(P.A, wA, d, nA)
    Bt = _batched_Bt(P.B, wB)
    Z = batched_mul(At, X) .+ reshape(Bt, d, 1, :)
    function vjp(cotangent)
        C = unthunk(cotangent)
        S = activation_derivative(σ, Z) .* C # d×N×K
        ΔX = batched_mul(batched_transpose(At), S) # d×N×K
        ΔAt = batched_mul(S, batched_transpose(X)) # d×d×K
        ΔP = zero(P)
        ΔP.A .= reshape(ΔAt, d, d, 1, :) .* reshape(wA, 1, 1, nA, 1)
        ΔP.B .= sum(S; dims=2) .* reshape(wB, 1, nB, 1)
        return (NoTangent(), ΔX, ΔP, NoTangent(), NoTangent(),
            NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return σ.(Z), vjp
end

# X0 = W_in·Ains
_embed_x0(P, Ains) = batched_mul(P.W_in, Ains)

function _bp_final(P, Ains, σ, repA, repB, d, nA, nB,
    solver, reltol, abstol, dt, sensealg)
    X0 = _embed_x0(P, Ains)
    rhs(u, q, t) = _f_flat_batched(u, q, t, σ, repA, repB, d, nA, nB)
    prob = ODEProblem(rhs, X0, (zero(eltype(X0)), one(eltype(X0))), P)
    sol = solve(prob, solver; reltol=reltol, abstol=abstol, _step_opts(dt)...,
        sensealg=sensealg, save_everystep=false)
    return sol.u[end]   # d×N×K
end

# Total loss summed over all K problems
function _bp_loss(P, Ains, Ystack, σ, repA, repB, d, nA, nB,
    solver, reltol, abstol, dt, sensealg, loss=squared_error_loss)
    Xf = _bp_final(P, Ains, σ, repA, repB, d, nA, nB,
        solver, reltol, abstol, dt, sensealg)
    return loss(batched_mul(P.W_out, Xf), Ystack)
end

# Per-problem E
function _bp_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB,
    solver, reltol, abstol, dt, loss=squared_error_loss)
    Xf = _bp_final(P, Ains, σ, repA, repB, d, nA, nB,
        solver, reltol, abstol, dt, _default_sensealg())
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, vec(sum(loss_per_sample(loss, Z, Ystack); dims=1)))
end

# Per-(sample, problem) error: like _bp_errors but keeps the sample axis. Returns an N×K matrix.
function _bp_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB,
    solver, reltol, abstol, dt, loss=squared_error_loss)
    Xf = _bp_final(P, Ains, σ, repA, repB, d, nA, nB,
        solver, reltol, abstol, dt, _default_sensealg())
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, loss_per_sample(loss, Z, Ystack))
end

"""
    batched_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB;
        solver=Tsit5(), reltol, abstol, dt, sensealg) → (E, ∇E)

Compute E(P) and its gradient for a whole same-shape group with the adjoint method.
"""
function batched_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB,
    d, nA, nB; solver=Tsit5(), reltol=1e-9, abstol=1e-12, dt=nothing,
    sensealg=_default_sensealg(), loss=squared_error_loss)
    E, gs = Zygote.withgradient(q ->
            _bp_loss(q, Ains, Ystack, σ, repA, repB, d, nA, nB,
                solver, reltol, abstol, dt, sensealg, loss), P)
    ∇E = gs[1]
    ∇E isa ComponentVector || (∇E = ComponentVector(∇E, getaxes(P)))
    return E, ∇E
end

# ==== Fast path ====
# A collection of hand written adjoints for speed.
_field(X, P, t, σ, repA, repB, d, nA, nB) =
    _f_flat_batched(X, P, t, σ, repA, repB, d, nA, nB)

_fast_method(backend::Symbol) = backend === :fasteuler ? :euler :
                                backend === :fasttsit5 ? :tsit5 :
                                backend === :fastrk4 ? :rk4 : nothing

# Solver + discrete adjoint
# Discards the trajectory and only returns the final state
function _fixedstep_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; method::Symbol=:rk4)
    X = _embed_x0(P, Ains)
    h = one(eltype(X)) / nsteps
    for s in 1:nsteps
        t = (s - 1) * h
        if method === :euler
            X = X .+ h .* _field(X, P, t, σ, repA, repB, d, nA, nB)
        elseif method === :tsit5
            X, _ = _t5_step(X, P, t, h, σ, repA, repB, d, nA, nB)
        else
            k1 = _field(X, P, t, σ, repA, repB, d, nA, nB)
            k2 = _field(X .+ (h / 2) .* k1, P, t + h / 2, σ, repA, repB, d, nA, nB)
            k3 = _field(X .+ (h / 2) .* k2, P, t + h / 2, σ, repA, repB, d, nA, nB)
            k4 = _field(X .+ h .* k3, P, t + h, σ, repA, repB, d, nA, nB)
            X = X .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
    end
    return X
end

fast_group_value(P, Ains, Ystack, σ, repA, repB, d, nA, nB; nsteps::Int=32, method::Symbol=:rk4, loss=squared_error_loss) =
    loss(batched_mul(P.W_out, _fixedstep_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; method)), Ystack)

# Per-problem E fast analog of _bp_errors
function _fast_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; method::Symbol=:rk4, loss=squared_error_loss)
    Xf = _fixedstep_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; method)
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, vec(sum(loss_per_sample(loss, Z, Ystack); dims=1)))
end

# Per-(sample, problem) error. Fast analog of _bp_sample_errors
function _fast_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; method::Symbol=:rk4, loss=squared_error_loss)
    Xf = _fixedstep_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; method)
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, loss_per_sample(loss, Z, Ystack))
end

"""
    fast_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB;
        nsteps=32, method=:rk4) → (E, ∇E)

Fast solver + discrete adjoint
"""
function fast_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB,
    d, nA, nB; nsteps::Int=32, method::Symbol=:rk4, loss=squared_error_loss)
    X0 = _embed_x0(P, Ains)
    h = one(eltype(X0)) / nsteps

    # Euler/RK4 share some structure. Tsit5 is separate.
    if method === :tsit5
        traj = Vector{typeof(X0)}(undef, nsteps + 1)
        traj[1] = X0
        for s in 1:nsteps
            X = traj[s]
            t = (s - 1) * h
            traj[s+1], _ = _t5_step(X, P, t, h, σ, repA, repB, d, nA, nB)
        end
    else
        field(X, t) = _f_flat_batched(X, P, t, σ, repA, repB, d, nA, nB)
        traj = _fixedstep_traj(X0, nsteps, h, field; method)
    end
    Xf = traj[nsteps+1]

    Z = batched_mul(P.W_out, Xf)
    E, dZ = loss_value_and_grad(loss, Z, Ystack)
    ∇E = zero(P)
    ∇E.W_out .= batched_mul(dZ, batched_transpose(Xf))
    costate = batched_mul(batched_transpose(P.W_out), dZ)

    if method === :tsit5
        for s in nsteps:-1:1
            X = traj[s]
            t = (s - 1) * h
            costate = _t5_step_adjoint!(∇E, costate, X, P, t, h, σ, repA, repB, d, nA, nB)
        end
    else
        vjp_at(X, t) = ChainRulesCore.rrule(_f_flat_batched, X, P, t, σ, repA, repB, d, nA, nB)
        costate = _fixedstep_adjoint!(∇E, costate, traj, nsteps, h, vjp_at; method)
    end
    ∇E.W_in .= batched_mul(costate, batched_transpose(Ains))
    return E, ∇E
end

# ==== Residual block fast ====
# Batched version of Adjoint.block_fast_value_and_gradient

"""
    batched_field_ops(repA, repB) → ops bundle for `build_block` (batched path).
"""
function batched_field_ops(repA, repB)
    return (
        basisA=(t, X) -> device_basis_weights(repA, t, X),
        basisB=(t, X) -> device_basis_weights(repB, t, X),
        weight=(P, m, wA) -> reshape(batched_vec(reshape(getproperty(P, Symbol(:A, m.idx)), m.wo * m.wi, length(wA), :), wA), m.wo, m.wi, :),
        bias=(P, m, wB) -> batched_vec(reshape(getproperty(P, Symbol(:B, m.idx)), m.wo, length(wB), :), wB),
        affine=(W, Y, b) -> batched_mul(W, Y) .+ reshape(b, size(b, 1), 1, size(b, 2)),
        tmul=(W, S) -> batched_mul(batched_transpose(W), S),
        (dweight!)=(ΔP, m, S, Y, wA) -> (getproperty(ΔP, Symbol(:A, m.idx)) .=
            reshape(batched_mul(S, batched_transpose(Y)), m.wo, m.wi, 1, :) .* reshape(wA, 1, 1, length(wA), 1)),
        (dbias!)=(ΔP, m, S, wB) -> (getproperty(ΔP, Symbol(:B, m.idx)) .=
            reshape(dropdims(sum(S; dims=2); dims=2), m.wo, 1, :) .* reshape(wB, 1, length(wB), 1)),
    )
end

"""
    block_stack_group(models::Vector{ResBlockFlow}) → P::ComponentVector

Stack a same-shape group into per-layer fields. Inverse of `block_unstack_group!`.
"""
function block_stack_group(models::AbstractVector{<:ResBlockFlow})
    spec = models[1].spec
    pv = [param_vector(m) for m in models]
    fields = Pair{Symbol,Any}[]
    for mm in spec.metas
        Al = cat((reshape(p.A[mm.rangeA], mm.wo, mm.wi, spec.ncp) for p in pv)...; dims=4)
        Bl = cat((reshape(p.B[mm.rangeB], mm.wo, spec.ncp) for p in pv)...; dims=3)
        push!(fields, Symbol(:A, mm.idx) => Al)
        push!(fields, Symbol(:B, mm.idx) => Bl)
    end
    push!(fields, :W_in => cat((m.W_in for m in models)...; dims=3))
    push!(fields, :W_out => cat((m.W_out for m in models)...; dims=3))
    return ComponentVector(NamedTuple(fields))
end

"""
    block_unstack_group!(models, P) → models

Write the k-th slice of the batched block params back into `models[k]`.
"""
function block_unstack_group!(models::AbstractVector{<:ResBlockFlow}, P)
    Pc = adapt(Array, P)
    spec = models[1].spec
    nL = length(spec.metas)
    for k in eachindex(models)
        Aflat = reduce(vcat, (vec(getproperty(Pc, Symbol(:A, l))[:, :, :, k]) for l in 1:nL))
        Bflat = reduce(vcat, (vec(getproperty(Pc, Symbol(:B, l))[:, :, k]) for l in 1:nL))
        set_params!(models[k], ComponentVector(A=Aflat, B=Bflat,
            W_in=Pc.W_in[:, :, k], W_out=Pc.W_out[:, :, k]))
    end
    return models
end

_block_eval_batched(model::ResBlockFlow) =
    build_block(model.spec, model.sigma, batched_field_ops(model.As[1], model.Bs[1]))

"""
    block_fast_group_value(P, Ains, Ystack, model; nsteps=32, method=:rk4) → E
"""
function block_fast_group_value(P, Ains, Ystack, model::ResBlockFlow;
    nsteps::Int=32, method::Symbol=:rk4, loss=squared_error_loss)
    be = _block_eval_batched(model)
    X0 = _embed_x0(P, Ains)
    h = one(eltype(X0)) / nsteps
    field(X, t) = first(be(X, P, t))
    Xf = _fixedstep_traj(X0, nsteps, h, field; method)[nsteps+1]
    return loss(batched_mul(P.W_out, Xf), Ystack)
end

"""
    block_fast_group_value_and_gradient(P, Ains, Ystack, model::ResBlockFlow;
        nsteps=32, method=:rk4) → (E, ∇E)

Batched adjoint value and gradient for the block model.
"""
function block_fast_group_value_and_gradient(P, Ains, Ystack, model::ResBlockFlow;
    nsteps::Int=32, method::Symbol=:rk4, loss=squared_error_loss)
    be = _block_eval_batched(model)
    X0 = _embed_x0(P, Ains)
    h = one(eltype(X0)) / nsteps
    field(X, t) = first(be(X, P, t))
    traj = _fixedstep_traj(X0, nsteps, h, field; method)
    Xf = traj[nsteps+1]

    Z = batched_mul(P.W_out, Xf)
    E, dZ = loss_value_and_grad(loss, Z, Ystack)
    ∇E = zero(P)
    ∇E.W_out .= batched_mul(dZ, batched_transpose(Xf))
    costate = batched_mul(batched_transpose(P.W_out), dZ)
    vjp_at(X, t) = (yp = be(X, P, t); (yp[1], cot -> (nothing, yp[2](cot)...)))
    costate = _fixedstep_adjoint!(∇E, costate, traj, nsteps, h, vjp_at; method)
    ∇E.W_in .= batched_mul(costate, batched_transpose(Ains))
    return E, ∇E
end

"""
    block_fast_group_errors(P, Ains, Ystack, model::ResBlockFlow; nsteps, method, loss)
        → Vector{Float64} (per-seed E)

Forward per-seed error. Block analog of `_fast_errors`.
"""
function block_fast_group_errors(P, Ains, Ystack, model::ResBlockFlow;
    nsteps::Int=32, method::Symbol=:rk4, loss=squared_error_loss)
    be = _block_eval_batched(model)
    X0 = _embed_x0(P, Ains)
    h = one(eltype(X0)) / nsteps
    field(X, t) = first(be(X, P, t))
    Xf = _fixedstep_traj(X0, nsteps, h, field; method)[nsteps+1]
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, vec(sum(loss_per_sample(loss, Z, Ystack); dims=1)))
end

"""
    block_fast_group_predict(P, Ains, model::ResBlockFlow; nsteps, method) → out×N×K (host)

Forward batched prediction `Z = W_out·x(1)`
"""
function block_fast_group_predict(P, Ains, model::ResBlockFlow;
    nsteps::Int=32, method::Symbol=:rk4)
    be = _block_eval_batched(model)
    X0 = _embed_x0(P, Ains)
    h = one(eltype(X0)) / nsteps
    field(X, t) = first(be(X, P, t))
    Xf = _fixedstep_traj(X0, nsteps, h, field; method)[nsteps+1]
    return adapt(Array, batched_mul(P.W_out, Xf))
end

"""
    train_block_batched(models::Vector{ResBlockFlow}, gens; backend=:fastrk4, nsteps=32,
        n_epochs=1000, num_samples=64, learning_rate=1e-3, data_to_device=identity, …)
        → (models, errors)

Train a same-shape group of block models as one batched solve.
"""
function train_block_batched(models::AbstractVector{<:ResBlockFlow}, gens;
    backend::Symbol=:fastrk4, nsteps::Int=32,
    n_epochs::Int=1000, num_samples::Int=64, learning_rate=1e-3,
    patience::Int=250, lr_decay=0.75, min_lr=1e-8, escape_frac::Float64=0.9,
    verbose::Int=10, data_to_device=identity, loss=squared_error_loss,
    on_epoch::Union{Nothing,Function}=nothing,
    checkpoint_every::Int=0, checkpoint_fn::Union{Nothing,Function}=nothing)
    length(models) == length(gens) || throw(ArgumentError("models and gens must have equal length"))
    method = _fast_method(backend)
    method === nothing && error("train_block_batched needs a fixed-step backend (:fastrk4/:fasteuler/:fasttsit5); got $backend")
    repr = models[1]
    ET = eltype(param_vector(repr))
    P = data_to_device(block_stack_group(models))
    opt = AdamFlat(P)

    fetch_batch(g) = (b = g(); (data_to_device(_stack(b[1], ET)), _stack(b[2], ET)))
    stack_ins(bs) = cat((b[1] for b in bs)...; dims=3)
    stack_tgts(bs) = data_to_device(cat((b[2] for b in bs)...; dims=3))

    eval_bs = [fetch_batch(g) for g in gens]
    Ains_eval = stack_ins(eval_bs)
    Ystack_eval = stack_tgts(eval_bs)

    ep = Ref(0)
    function step!(lr)
        bs = [fetch_batch(g) for g in gens]
        Ains = stack_ins(bs)
        Ystack = stack_tgts(bs)
        _, ∇E = block_fast_group_value_and_gradient(P, Ains, Ystack, repr; nsteps, method, loss)
        apply_gradients!(opt, P, lr, ∇E)
        ep[] += 1
        # Checkpoint: unstack the params into models and pass them to checkpoint_fn.
        if checkpoint_fn !== nothing && checkpoint_every > 0 && ep[] % checkpoint_every == 0
            block_unstack_group!(models, P)
            checkpoint_fn(models)
        end
        return block_fast_group_value(P, Ains_eval, Ystack_eval, repr; nsteps, method, loss)
    end

    errors = train_loop(step!, n_epochs, num_samples; learning_rate, patience,
        lr_decay, min_lr, escape_frac, verbose, label="Block", on_epoch)
    block_unstack_group!(models, P)
    return models, errors
end

# ==== Fast adaptive Tsit5  ====
# Like the fast RK4 path, but Tsit5 picks the step sizes.
#
# Tsit5 tableau (Tsitouras 2011)
const _T5_C = (0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0)
const _T5_A = (
    (),
    (0.161,),
    (-0.008480655492356989, 0.335480655492357),
    (2.8971530571054935, -6.359448489975075, 4.3622954328695815),
    (5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525),
    (5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383),
)
const _T5_B = (0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774)
const _T5_BT = (-0.00178001105222577714, -0.0008164344596567469, 0.007880878010261995,
    -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 0.015151515151515152)

# RMS norm over the whole batched array, scaled by the per-element tolerance.
_t5_errnorm(errest, X, Xnew, reltol, abstol) = begin
    T = eltype(X)
    sc = T(abstol) .+ T(reltol) .* max.(abs.(X), abs.(Xnew))
    sqrt(sum(abs2, errest ./ sc) / length(errest))
end

# One Tsit5 step from (X, t) of size h
function _t5_step(X, P, t, h, σ, repA, repB, d, nA, nB)
    T = eltype(X)
    ks = Vector{typeof(X)}(undef, 6)
    for i in 1:6
        Yi = X
        ai = _T5_A[i]
        for j in 1:(i-1)
            Yi = Yi .+ (h * T(ai[j])) .* ks[j]
        end
        ks[i] = _field(Yi, P, t + T(_T5_C[i]) * h, σ, repA, repB, d, nA, nB)
    end
    Xnew = X
    for i in 1:6
        Xnew = Xnew .+ (h * T(_T5_B[i])) .* ks[i]
    end
    k7 = _field(Xnew, P, t + h, σ, repA, repB, d, nA, nB)
    errest = (h * T(_T5_BT[7])) .* k7
    for i in 1:6
        errest = errest .+ (h * T(_T5_BT[i])) .* ks[i]
    end
    return Xnew, errest
end

# Adjoint of one Tsit5 step
function _t5_step_adjoint!(∇E, costate, X, P, t, h, σ, repA, repB, d, nA, nB)
    T = eltype(X)
    ks = Vector{typeof(X)}(undef, 6)
    vs = Vector{Any}(undef, 6)
    for i in 1:6
        Yi = X
        ai = _T5_A[i]
        for j in 1:(i-1)
            Yi = Yi .+ (h * T(ai[j])) .* ks[j]
        end
        ks[i], vs[i] = ChainRulesCore.rrule(_f_flat_batched, Yi, P,
            t + T(_T5_C[i]) * h, σ, repA, repB, d, nA, nB)
    end
    costate_prev = copy(costate)
    Δk = [(h * T(_T5_B[i])) .* costate for i in 1:6]   # b_i contribution
    for i in 6:-1:1
        r = vs[i](Δk[i])
        ∇E.A .+= r[3].A
        ∇E.B .+= r[3].B
        ΔYi = r[2]
        costate_prev .+= ΔYi
        ai = _T5_A[i]
        for j in 1:(i-1)
            Δk[j] .+= (h * T(ai[j])) .* ΔYi
        end
    end
    return costate_prev
end

function _t5_initial_dt(X0, P, σ, repA, repB, d, nA, nB, reltol, abstol)
    T = eltype(X0)
    sc = T(abstol) .+ T(reltol) .* abs.(X0)
    d0 = sqrt(sum(abs2, X0 ./ sc) / length(X0))
    f0 = _field(X0, P, zero(T), σ, repA, repB, d, nA, nB)
    d1 = sqrt(sum(abs2, f0 ./ sc) / length(X0))
    h0 = (d0 < 1e-5 || d1 < 1e-5) ? T(1e-6) : T(0.01 * (d0 / d1))
    return min(h0, T(0.1))
end

# Adaptive solve over [0,1]. returns (traj, ts, hs)
function _t5_schedule(X0, P, σ, repA, repB, d, nA, nB; reltol, abstol, maxsteps::Int=100_000)
    T = eltype(X0)
    tend = one(T)
    t = zero(T)
    X = X0
    h = _t5_initial_dt(X0, P, σ, repA, repB, d, nA, nB, reltol, abstol)
    traj = [X0]
    ts = T[]
    hs = T[]
    steps = 0
    while t < tend && steps < maxsteps
        h = min(h, tend - t)
        Xnew, errest = _t5_step(X, P, t, h, σ, repA, repB, d, nA, nB)
        en = _t5_errnorm(errest, X, Xnew, reltol, abstol)
        steps += 1
        # Step-size controller
        fac = en == 0 ? 10.0 : clamp(0.9 * en^(-0.2), 0.2, 10.0)
        if en <= 1 || h <= T(1e-12)
            push!(ts, t)
            push!(hs, h)
            t += h
            X = Xnew
            push!(traj, X)
            h = h * T(fac)
        else
            h = h * T(min(fac, 1.0))
        end
    end
    return traj, ts, hs
end

function _fastadaptive_final(P, Ains, σ, repA, repB, d, nA, nB; reltol, abstol)
    X0 = _embed_x0(P, Ains)
    traj, _, _ = _t5_schedule(X0, P, σ, repA, repB, d, nA, nB; reltol, abstol)
    return traj[end]
end

# Forward loss
fastadaptive_group_value(P, Ains, Ystack, σ, repA, repB, d, nA, nB; reltol=1e-3, abstol=1e-6, loss=squared_error_loss) =
    loss(batched_mul(P.W_out, _fastadaptive_final(P, Ains, σ, repA, repB, d, nA, nB; reltol, abstol)), Ystack)

function _fastadaptive_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB; reltol, abstol, loss=squared_error_loss)
    Xf = _fastadaptive_final(P, Ains, σ, repA, repB, d, nA, nB; reltol, abstol)
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, vec(sum(loss_per_sample(loss, Z, Ystack); dims=1)))
end

function _fastadaptive_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB; reltol, abstol, loss=squared_error_loss)
    Xf = _fastadaptive_final(P, Ains, σ, repA, repB, d, nA, nB; reltol, abstol)
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, loss_per_sample(loss, Z, Ystack))
end

"""
    fastadaptive_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB;
        reltol=1e-3, abstol=1e-6) → (E, ∇E)

Adaptive Tsit5 forward
"""
function fastadaptive_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB,
    d, nA, nB; reltol=1e-3, abstol=1e-6, loss=squared_error_loss)
    X0 = _embed_x0(P, Ains)
    traj, ts, hs = _t5_schedule(X0, P, σ, repA, repB, d, nA, nB; reltol, abstol)
    Xf = traj[end]

    Z = batched_mul(P.W_out, Xf)
    E, dZ = loss_value_and_grad(loss, Z, Ystack)
    ∇E = zero(P)
    ∇E.W_out .= batched_mul(dZ, batched_transpose(Xf))
    costate = batched_mul(batched_transpose(P.W_out), dZ)
    for s in length(hs):-1:1
        costate = _t5_step_adjoint!(∇E, costate, traj[s], P, ts[s], hs[s], σ, repA, repB, d, nA, nB)
    end
    ∇E.W_in .= batched_mul(costate, batched_transpose(Ains))
    return E, ∇E
end

# ==== Forward Picard path ====
# Picard iteration on a fixed node grid. Uses Zygote's AD because I couldn't figure this one out.
const _PICARD_ITERS = 30

function _picard_weights(M::Int, ::Type{T}) where {T}
    h = one(T) / (M - 1)
    W = zeros(T, M, M)
    for m in 2:M, j in 1:m
        W[m, j] = (j == 1 || j == m) ? h / 2 : h
    end
    return W
end
# Weights depend only on the grid, not P.
ChainRulesCore.@non_differentiable _picard_weights(::Any, ::Any)

function _picard_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; piters::Int=_PICARD_ITERS)
    X0 = _embed_x0(P, Ains)
    T = eltype(X0)
    M = nsteps + 1
    h = one(T) / nsteps
    ts = [(m - 1) * h for m in 1:M]
    W = _picard_weights(M, T)
    traj = [X0 for _ in 1:M]
    for _ in 1:piters
        F = [_field(traj[m], P, ts[m], σ, repA, repB, d, nA, nB) for m in 1:M]
        traj = [X0 .+ sum(W[m, j] .* F[j] for j in 1:m) for m in 1:M]
    end
    return traj[M]
end

# Forward loss
picard_group_value(P, Ains, Ystack, σ, repA, repB, d, nA, nB; nsteps::Int, piters::Int=_PICARD_ITERS, loss=squared_error_loss) =
    loss(batched_mul(P.W_out, _picard_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; piters)), Ystack)

# Per-problem E. Picard analog of _fast_errors.
function _picard_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; piters::Int=_PICARD_ITERS, loss=squared_error_loss)
    Xf = _picard_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; piters)
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, vec(sum(loss_per_sample(loss, Z, Ystack); dims=1)))
end

# Per-(sample, problem) error. Picard analog of _fast_sample_errors.
function _picard_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; piters::Int=_PICARD_ITERS, loss=squared_error_loss)
    Xf = _picard_final(P, Ains, σ, repA, repB, d, nA, nB, nsteps; piters)
    Z = batched_mul(P.W_out, Xf)
    return adapt(Array, loss_per_sample(loss, Z, Ystack))
end

"""
    picard_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB;
        nsteps, piters=_PICARD_ITERS) → (E, ∇E)

Picard solve + Zygote AD.
"""
function picard_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB,
    d, nA, nB; nsteps::Int, piters::Int=_PICARD_ITERS, loss=squared_error_loss)
    E, gs = Zygote.withgradient(q ->
            loss(batched_mul(q.W_out,
                    _picard_final(q, Ains, σ, repA, repB, d, nA, nB, nsteps; piters)), Ystack), P)
    ∇E = gs[1]
    ∇E isa ComponentVector || (∇E = ComponentVector(∇E, getaxes(P)))
    return E, ∇E
end

# ==== Training ====
function _train_group(models, gens; solver, n_epochs, learning_rate, num_samples,
    patience, lr_decay, min_lr, verbose, data_to_device, reltol, abstol, dt,
    sensealg, eval_every, sync_every, on_epoch, on_loss, backend, nsteps,
    loss=squared_error_loss,
    collect_sample_errors=false,
    per_seed_schedule::Bool=false, min_improve::Float64=0.0, plateau_evals::Int=0,
    escape_frac::Float64=0.9, train_win::Bool=false,
    lr0::Union{Nothing,AbstractVector}=nothing,
    lr_out::Union{Nothing,AbstractVector}=nothing,
    on_eval::Union{Nothing,Function}=nothing)

    σ = models[1].sigma
    d = length(models[1].B[1])
    nA = length(models[1].A)
    nB = length(models[1].B)
    K = length(models)

    isadaptive = backend === :fastadaptive
    ispicard = backend === :picard
    fastmethod = _fast_method(backend)
    isfast = fastmethod !== nothing

    P = data_to_device(stack_group(models))

    sensealg === nothing && (sensealg = _auto_sensealg(P))

    optP = AdamFlat(P)

    ET = eltype(P)
    fetch_batch(g) = begin
        ins, tgts = g()
        (data_to_device(_stack(ins, ET)), _stack(tgts, ET))
    end
    stack_ins(bs) = cat((b[1] for b in bs)...; dims=3)
    stack_tgts(bs) = data_to_device(cat((b[2] for b in bs)...; dims=3))

    eval_batches = [fetch_batch(g) for g in gens]
    Ains_eval = stack_ins(eval_batches)
    Ystack_eval = stack_tgts(eval_batches)
    X0_example = _embed_x0(P, Ains_eval)
    repA = BasisCache(models[1].A, X0_example)
    repB = BasisCache(models[1].B, X0_example)

    spawn_batch() = Threads.@spawn begin
        bs = [fetch_batch(g) for g in gens]
        (stack_ins(bs), stack_tgts(bs))
    end
    next_batch = spawn_batch()

    errs = [Float64[] for _ in 1:K]
    sample_errs = [Vector{Float64}[] for _ in 1:K]
    iters = Ref(0)
    cur_lr = Ref(0.0)

    # Per-seed LR schedule
    lr_host = lr0 === nothing ? fill(Float64(learning_rate), K) : collect(Float64, lr0)
    lr_dev = Ref(per_seed_schedule ? data_to_device(ET.(lr_host)) : nothing)
    best_k = fill(Inf, K)
    stale_k = zeros(Int, K)
    frozen = lr0 === nothing ? falses(K) : BitVector(lr_host[k] == 0.0 for k in 1:K)
    patience_checks = max(1, round(Int, patience / max(eval_every, 1)))
    allfrozen = Ref(false)
    # Escape gate: a seed holds its LR at base until its error drops below escape_frac*(first error)
    e_first_k = fill(NaN, K)
    escaped_k = escape_frac >= 1 ? trues(K) :
                lr0 === nothing ? falses(K) :
                BitVector(lr_host[k] < Float64(learning_rate) for k in 1:K)
    function update_perseed!(perseed_err)
        changed = false
        nfroz = 0
        for k in 1:K
            if frozen[k]
                nfroz += 1
                continue
            end
            isnan(e_first_k[k]) && (e_first_k[k] = perseed_err[k])
            if perseed_err[k] < best_k[k] * (1 - min_improve)
                best_k[k] = perseed_err[k]
                stale_k[k] = 0
            else
                stale_k[k] += 1
            end
            if !escaped_k[k]
                if best_k[k] < escape_frac * e_first_k[k]
                    escaped_k[k] = true
                else
                    stale_k[k] = 0
                    continue
                end
            end
            if lr_host[k] > min_lr
                if stale_k[k] >= patience_checks
                    lr_host[k] = max(lr_host[k] * lr_decay, min_lr)
                    stale_k[k] = 0
                    changed = true
                end
            elseif plateau_evals > 0 && stale_k[k] >= plateau_evals
                frozen[k] = true
                lr_host[k] = 0.0
                nfroz += 1
                changed = true
            end
        end
        changed && (lr_dev[] = data_to_device(ET.(lr_host)))
        allfrozen[] = nfroz == K
        return nothing
    end

    function step!(lr)
        cur_lr[] = per_seed_schedule ? sum(lr_host) / K : lr
        Ains, Ystack = fetch(next_batch)
        next_batch = spawn_batch()
        E, ∇E = ispicard ?
                picard_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB; nsteps, loss) :
                isadaptive ?
                fastadaptive_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB; reltol, abstol, loss) :
                isfast ?
                fast_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB; nsteps, method=fastmethod, loss) :
                batched_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, d, nA, nB;
            solver, reltol, abstol, dt, sensealg, loss)
        if per_seed_schedule
            apply_gradients_perseed!(optP, P, lr_dev[], ∇E; train_win)
            lr_out !== nothing && copyto!(lr_out, lr_host)
        else
            apply_gradients!(optP, P, lr, ∇E)
        end
        iters[] += 1
        if eval_every > 0 && iters[] % eval_every == 0
            if collect_sample_errors
                S = ispicard ?
                    _picard_sample_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, nsteps; loss) :
                    isadaptive ?
                    _fastadaptive_sample_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB; reltol, abstol, loss) :
                    isfast ?
                    _fast_sample_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, nsteps; method=fastmethod, loss) :
                    _bp_sample_errors(P, Ains_eval, Ystack_eval, σ, repA, repB,
                    d, nA, nB, solver, reltol, abstol, dt, loss)
                tot = 0.0
                perseed = Vector{Float64}(undef, K)
                for k in 1:K
                    col = Float64.(@view S[:, k])
                    push!(sample_errs[k], col)
                    sk = sum(col)
                    push!(errs[k], sk)
                    perseed[k] = sk
                    tot += sk
                end
                on_eval !== nothing && on_eval(iters[], perseed)
                per_seed_schedule && update_perseed!(perseed)
                return tot
            end
            ek = ispicard ?
                 _picard_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, nsteps; loss) :
                 isadaptive ?
                 _fastadaptive_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB; reltol, abstol, loss) :
                 isfast ?
                 _fast_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, nsteps; method=fastmethod, loss) :
                 _bp_errors(P, Ains_eval, Ystack_eval, σ, repA, repB,
                d, nA, nB, solver, reltol, abstol, dt, loss)
            perseed = Float64[ek[k] for k in 1:K]
            for k in 1:K
                push!(errs[k], ek[k])
            end
            on_eval !== nothing && on_eval(iters[], perseed)
            per_seed_schedule && update_perseed!(perseed)
            return sum(ek)
        end
        if eval_every == 0 && iters[] % sync_every == 0
            ek = ispicard ?
                 _picard_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, nsteps; loss) :
                 isadaptive ?
                 _fastadaptive_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB; reltol, abstol, loss) :
                 isfast ?
                 _fast_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, nsteps; method=fastmethod, loss) :
                 _bp_errors(P, Ains_eval, Ystack_eval, σ, repA, repB, d, nA, nB, solver, reltol, abstol, dt, loss)
            for k in 1:K
                push!(errs[k], ek[k])
            end
        end
        return E
    end

    # on_loss is cheap. on_epoch needs the unstacked models, so it needs to copy from gpu
    wrapped = (on_epoch === nothing && on_loss === nothing && !per_seed_schedule) ? nothing :
              (E, epoch, errors) -> begin
        on_loss === nothing || on_loss(E, epoch, cur_lr[])
        per_seed_schedule && allfrozen[] && return :stop   # every seed frozen ⇒ done
        on_epoch === nothing && return nothing
        epoch % sync_every == 0 || return nothing
        unstack_group!(models, P)
        on_epoch(E, epoch, errors)
    end

    label = "Batched($(nameof(typeof(solver))), K=$K)"

    train_loop(step!, n_epochs, num_samples;
        learning_rate, verbose, patience=(per_seed_schedule ? n_epochs + 1 : patience),
        lr_decay, min_lr, escape_frac=(per_seed_schedule ? 2.0 : escape_frac),
        label, on_epoch=wrapped)

    unstack_group!(models, P)
    return errs, sample_errs
end

function _chunk_ranges(n::Int, nchunks::Int)
    nchunks = clamp(nchunks, 1, n)
    base, rem = divrem(n, nchunks)
    ranges = Vector{UnitRange{Int}}(undef, nchunks)
    start = 1
    for c in 1:nchunks
        len = base + (c <= rem ? 1 : 0)
        ranges[c] = start:(start+len-1)
        start += len
    end
    return ranges
end

#On CPU split the the group across threads
function _blas_thread_safe()
    omp = tryparse(Int, get(ENV, "OMP_NUM_THREADS", ""))
    omp === nothing && return false
    return Threads.nthreads() * omp <= 64
end

function _train_group_threaded(models, gens; data_to_device, verbose, on_epoch, on_loss,
    lr0::Union{Nothing,AbstractVector}=nothing,
    lr_out::Union{Nothing,AbstractVector}=nothing,
    on_eval::Union{Nothing,Function}=nothing, kwargs...)
    cpu = data_to_device === identity
    nchunks = (cpu && _blas_thread_safe()) ? min(Threads.nthreads(), length(models)) : 1
    if nchunks <= 1
        if cpu && Threads.nthreads() > 1 && !_blas_thread_safe()
            @warn "CPU training running single-threaded: set OMP_NUM_THREADS=1 in the \
                   launch environment to thread seeds across cores safely. \
                   Threading with the current OpenBLAS thread count would corrupt the heap." maxlog = 1
        end
        return _train_group(models, gens; data_to_device, verbose, on_epoch, on_loss,
            lr0, lr_out, on_eval, kwargs...)
    end

    ranges = _chunk_ranges(length(models), nchunks)
    g_errs = Vector{Vector{Float64}}(undef, length(models))
    g_sample = Vector{Vector{Vector{Float64}}}(undef, length(models))
    blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        @sync for (c, rng) in enumerate(ranges)
            Threads.@spawn begin

                e, se = _train_group(models[rng], gens[rng]; data_to_device,
                    verbose=c == 1 ? verbose : 0,
                    on_epoch=c == 1 ? on_epoch : nothing,
                    on_loss=c == 1 ? on_loss : nothing,
                    lr0=lr0 === nothing ? nothing : @view(lr0[rng]),
                    lr_out=lr_out === nothing ? nothing : @view(lr_out[rng]),
                    kwargs...)
                for (j, k) in enumerate(rng)
                    g_errs[k] = e[j]
                    g_sample[k] = se[j]
                end
            end
        end
    finally
        BLAS.set_num_threads(blas)
    end
    return g_errs, g_sample
end

"""
    train_batched(models, gens; solver=Tsit5(), n_epochs, learning_rate,
        num_samples, data_to_device=identity, ...) → (models, errors, sample_errors)

Train a set of independent problems, grouping same-shape ones into batched
solves
"""
function train_batched(models::AbstractVector, gens::AbstractVector;
    solver=Tsit5(), n_epochs=100, learning_rate=1e-3, num_samples=128,
    patience=250, lr_decay=0.75, min_lr=1e-8, verbose=10,
    data_to_device=identity, reltol=1e-9, abstol=1e-12, dt=nothing,
    sensealg=nothing, eval_every::Int=1, sync_every::Int=1,
    on_epoch::Union{Nothing,Function}=nothing, on_loss::Union{Nothing,Function}=nothing,
    backend::Symbol=:sciml, nsteps::Int=32, loss=squared_error_loss,
    collect_sample_errors::Bool=false,
    per_seed_schedule::Bool=false, min_improve::Float64=0.0, plateau_evals::Int=0,
    escape_frac::Float64=0.9, train_win::Bool=false,
    lr0::Union{Nothing,AbstractVector}=nothing,
    lr_out::Union{Nothing,AbstractVector}=nothing,
    on_eval::Union{Nothing,Function}=nothing)

    groups = group_by_shape(models, gens)
    errors = Vector{Vector{Float64}}(undef, length(models))

    sample_errors = Vector{Vector{Vector{Float64}}}(undef, length(models))
    for grp in groups
        g_errs, g_sample = _train_group_threaded(grp.models, grp.gens; solver, n_epochs,
            learning_rate, num_samples, patience, lr_decay, min_lr, verbose,
            data_to_device, reltol, abstol, dt, sensealg, eval_every, sync_every, on_epoch,
            on_loss, backend, nsteps, loss, collect_sample_errors,
            per_seed_schedule, min_improve, plateau_evals, escape_frac, train_win, on_eval,
            lr0=lr0 === nothing ? nothing : view(lr0, grp.idx),
            lr_out=lr_out === nothing ? nothing : view(lr_out, grp.idx))
        for (local_k, orig_i) in enumerate(grp.idx)
            models[orig_i] = grp.models[local_k]
            errors[orig_i] = g_errs[local_k]
            sample_errors[orig_i] = g_sample[local_k]
        end
    end
    # Record per-seed loss curves
    for i in eachindex(errors)
        record_curve!(errors[i]; tag=i, kind=:per_seed)
    end
    return models, errors, sample_errors
end

"""
    batched_heldout_errors(models, gens; backend=:sciml, nsteps=32, solver=Tsit5(),
        reltol=1e-9, abstol=1e-12, dt=nothing, data_to_device=identity) → Vector{Float64}

Per-sample error for each model
"""
function batched_heldout_errors(models::AbstractVector, gens::AbstractVector;
    backend::Symbol=:sciml, nsteps::Int=32, solver=Tsit5(),
    reltol=1e-9, abstol=1e-12, dt=nothing, data_to_device=identity, loss=squared_error_loss)

    isadaptive = backend === :fastadaptive
    ispicard = backend === :picard
    fastmethod = _fast_method(backend)
    isfast = fastmethod !== nothing
    out = Vector{Float64}(undef, length(models))
    for grp in group_by_shape(models, gens)
        ms = grp.models
        σ = ms[1].sigma
        d = length(ms[1].B[1])
        nA = length(ms[1].A)
        nB = length(ms[1].B)
        P = data_to_device(stack_group(ms))
        ET = eltype(P)
        batches = [g() for g in grp.gens]
        Ains = data_to_device(cat((_stack(ins, ET) for (ins, _) in batches)...; dims=3))
        Ystack = data_to_device(cat((_stack(tgts, ET) for (_, tgts) in batches)...; dims=3))
        n = size(Ystack, 2)
        X0 = _embed_x0(P, Ains)
        repA = BasisCache(ms[1].A, X0)
        repB = BasisCache(ms[1].B, X0)
        ek = ispicard ?
             _picard_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; loss) :
             isadaptive ?
             _fastadaptive_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB; reltol, abstol, loss) :
             isfast ?
             _fast_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; method=fastmethod, loss) :
             _bp_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, solver, reltol, abstol, dt, loss)
        for (lk, oi) in enumerate(grp.idx)
            out[oi] = ek[lk] / n
        end
    end
    return out
end

"""
    batched_heldout_sample_errors(models, gens; backend=:sciml, nsteps=32, solver=Tsit5(),
        reltol=1e-9, abstol=1e-12, dt=nothing, data_to_device=identity) → Vector{Vector{Float64}}

Like `batched_heldout_errors` but keeps the sample axis.
"""
function batched_heldout_sample_errors(models::AbstractVector, gens::AbstractVector;
    backend::Symbol=:sciml, nsteps::Int=32, solver=Tsit5(),
    reltol=1e-9, abstol=1e-12, dt=nothing, data_to_device=identity, loss=squared_error_loss)

    isadaptive = backend === :fastadaptive
    ispicard = backend === :picard
    fastmethod = _fast_method(backend)
    isfast = fastmethod !== nothing
    out = Vector{Vector{Float64}}(undef, length(models))
    for grp in group_by_shape(models, gens)
        ms = grp.models
        σ = ms[1].sigma
        d = length(ms[1].B[1])
        nA = length(ms[1].A)
        nB = length(ms[1].B)
        P = data_to_device(stack_group(ms))
        ET = eltype(P)
        batches = [g() for g in grp.gens]
        Ains = data_to_device(cat((_stack(ins, ET) for (ins, _) in batches)...; dims=3))
        Ystack = data_to_device(cat((_stack(tgts, ET) for (_, tgts) in batches)...; dims=3))
        X0 = _embed_x0(P, Ains)
        repA = BasisCache(ms[1].A, X0)
        repB = BasisCache(ms[1].B, X0)
        S = ispicard ?
            _picard_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; loss) :
            isadaptive ?
            _fastadaptive_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB; reltol, abstol, loss) :
            isfast ?
            _fast_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, nsteps; method=fastmethod, loss) :
            _bp_sample_errors(P, Ains, Ystack, σ, repA, repB, d, nA, nB, solver, reltol, abstol, dt, loss)
        for (lk, oi) in enumerate(grp.idx)
            out[oi] = Float64.(@view S[:, lk])
        end
    end
    return out
end

end
