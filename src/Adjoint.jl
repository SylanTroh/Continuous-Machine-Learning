"""
This file computes the gradient ∇E(P) with the adjoint method using SciMLSensitivity.

References:
- [Gho19] Gholami, Keutzer & Biros, "ANODE: Unconditionally Accurate
  Memory-Efficient Gradients for Neural ODEs", IJCAI 2019.
- [Rack20] Rackauckas, "Parallel Computing and Scientific Machine Learning",
  MIT 18.337 notes, lectures 10-11. https://book.sciml.ai/
- [SciML] SciMLSensitivity.jl, "Sensitivity Math Details".
  https://docs.sciml.ai/SciMLSensitivity/stable/sensitivity_math/
"""
module Adjoint

using LinearAlgebra
using Adapt
using ChainRulesCore: ChainRulesCore, NoTangent
using ComponentArrays
using DifferentialEquations
using SciMLSensitivity
using Zygote
using ..AbstractRepresentation
using ..SplineRepresentation
using ..ActivationFunctions
using ..ODEModel
using ..BlockField: build_block, cpu_field_ops
using ..ResBlockModel: ResBlockFlow
using ..GradientUtils
using ..TrainingUtils: compute_error
using ..ModelInit: init_model
using ..Tasks: DeterminantTask, generate
using ..ModelUtils: copy_model, param_vector, set_params!
using ..Results: Checkpointer, on_improve!, checkpoint!, load_run

export adjoint_gradients, adjoint_value_and_gradients
export batched_value_and_gradient, batched_predict
export fast_value_and_gradient, fast_value
export block_fast_value, block_fast_value_and_gradient
export train_adjoint, train_determinant_adjoint
export squared_error_loss, softmax_crossentropy_loss

_stack(xs, ::Type{ET}) where {ET} = ET.(reduce(hcat, vec.(xs)))

"""
Sum of squares
"""
squared_error_loss(Z, Y) = sum(abs2, Z .- Y)

"""
Softmax cross entropy
"""
function softmax_crossentropy_loss(Z, Y)
    #Apply a shift so that eᶻ doesn't blow up. The ms cancel out so this does not affect the derivative.
    m = ChainRulesCore.ignore_derivatives(maximum(Z; dims=1))
    lse = m .+ log.(sum(exp.(Z .- m); dims=1))
    return sum(lse) - sum(Y .* Z)
end

"""
Returns (E, ∂E/∂Z) where Z = W_out ⋅ x(1)
Used to calculate the initial condition for the backwards solve
"""
loss_value_and_grad(::typeof(squared_error_loss), Z, Y) = (sum(abs2, Z .- Y), 2 .* (Z .- Y))
function loss_value_and_grad(::typeof(softmax_crossentropy_loss), Z, Y)
    m = maximum(Z; dims=1)
    e = exp.(Z .- m)
    s = sum(e; dims=1)
    val = sum(m .+ log.(s)) - sum(Y .* Z)
    return val, e ./ s .- Y
end

"""
Evaluate f using a flat vector p = (A, B, W_in, W_out)
so Zygote can differentiate it.
"""
function _f_flat(X, p, t, σ, repA, repB, d, nA)
    wA = device_basis_weights(repA, t, X)
    wB = device_basis_weights(repB, t, X)
    At = reshape(reshape(p.A, d * d, nA) * wA, d, d)
    Bt = p.B * wB
    return σ.(At * X .+ Bt)
end

"""
RRule for _f_flat
"""
function ChainRulesCore.rrule(::typeof(_f_flat), X, p, t, σ, repA, repB, d, nA)
    wA = device_basis_weights(repA, t, X)
    wB = device_basis_weights(repB, t, X)
    #A(t) and B(t)
    At = reshape(reshape(p.A, d * d, nA) * wA, d, d)
    Bt = p.B * wB
    Z = At * X .+ Bt
    function vjp(cotangent)
        S = activation_derivative(σ, Z) .* cotangent
        ΔX = At' * S
        Δp = zero(p)
        Δp.A .= reshape(vec(S * X') * wA', d, d, nA)
        Δp.B .= sum(S; dims=2) * wB'
        return (NoTangent(), ΔX, Δp, NoTangent(), NoTangent(),
            NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return σ.(Z), vjp
end

"""
Fixed step forward solver
Returns the full trajectory
"""
function _fixedstep_traj(X0, nsteps, h, field; method::Symbol=:rk4)
    traj = Vector{typeof(X0)}(undef, nsteps + 1)
    traj[1] = X0
    for s in 1:nsteps
        X = traj[s]
        t = (s - 1) * h
        if method === :euler
            traj[s+1] = X .+ h .* field(X, t)
        else
            k1 = field(X, t)
            k2 = field(X .+ (h / 2) .* k1, t + h / 2)
            k3 = field(X .+ (h / 2) .* k2, t + h / 2)
            k4 = field(X .+ h .* k3, t + h)
            traj[s+1] = X .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
    end
    return traj
end

_accum_params!(g, delta) = (ComponentArrays.getdata(g) .+= ComponentArrays.getdata(delta); g)

"""
Fixed step backward solver
Uses the adjoint method
"""
function _fixedstep_adjoint!(g, costate, traj, nsteps, h, vjp_at; method::Symbol=:rk4)
    for s in nsteps:-1:1
        X = traj[s]
        t = (s - 1) * h
        if method === :euler
            # xₙ = X + h·f(X, t)
            _, v1 = vjp_at(X, t)
            costate_prev = copy(costate)
            r1 = v1(h .* costate)
            _accum_params!(g, r1[3])
            costate_prev .+= r1[2]
            costate = costate_prev
        else
            k1, v1 = vjp_at(X, t)
            y2 = X .+ (h / 2) .* k1
            k2, v2 = vjp_at(y2, t + h / 2)
            y3 = X .+ (h / 2) .* k2
            k3, v3 = vjp_at(y3, t + h / 2)
            y4 = X .+ h .* k3
            _, v4 = vjp_at(y4, t + h)
            # xₙ = X + h/6·(k1 + 2k2 + 2k3 + k4)
            costate_prev = copy(costate)
            Δk1 = (h / 6) .* costate
            Δk2 = (h / 3) .* costate
            Δk3 = (h / 3) .* costate
            Δk4 = (h / 6) .* costate
            r4 = v4(Δk4)
            _accum_params!(g, r4[3])
            costate_prev .+= r4[2]
            Δk3 .+= h .* r4[2]
            r3 = v3(Δk3)
            _accum_params!(g, r3[3])
            costate_prev .+= r3[2]
            Δk2 .+= (h / 2) .* r3[2]
            r2 = v2(Δk2)
            _accum_params!(g, r2[3])
            costate_prev .+= r2[2]
            Δk1 .+= (h / 2) .* r2[2]
            r1 = v1(Δk1)
            _accum_params!(g, r1[3])
            costate_prev .+= r1[2]
            costate = costate_prev
        end
    end
    return costate
end

"""
Use the (faster) cached basis functions instead of the parameters if they exist
"""
_fast_reps(model, basis_caches) = basis_caches === nothing ? (model.A, model.B) : basis_caches

"""
Fixed step forward solver
Discards the trajectory and only returns the final state
"""
function _fixedstep_final(p, model, Ain, nsteps, repA, repB; method::Symbol=:rk4)
    σ = model.sigma
    d = length(model.B[1])
    nA = length(model.A)
    X = p.W_in * Ain
    h = one(eltype(X)) / nsteps
    for s in 1:nsteps
        t = (s - 1) * h
        if method === :euler
            X = X .+ h .* _f_flat(X, p, t, σ, repA, repB, d, nA)
        else
            k1 = _f_flat(X, p, t, σ, repA, repB, d, nA)
            k2 = _f_flat(X .+ (h / 2) .* k1, p, t + h / 2, σ, repA, repB, d, nA)
            k3 = _f_flat(X .+ (h / 2) .* k2, p, t + h / 2, σ, repA, repB, d, nA)
            k4 = _f_flat(X .+ h .* k3, p, t + h, σ, repA, repB, d, nA)
            X = X .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
    end
    return X
end

"""
Evaluate the model without saving the trajectory
"""
fast_value(p, model, Ain, Y; nsteps::Int=32, basis_caches=nothing, method::Symbol=:rk4,
    loss=squared_error_loss) =
    (rs = _fast_reps(model, basis_caches);
    loss(p.W_out * _fixedstep_final(p, model, Ain, nsteps, rs[1], rs[2]; method), Y))

"""
    fast_value_and_gradient(p, model, Ain, Y; nsteps=32, basis_caches=nothing, method=:rk4) → (E, ∇E)

Evaluate the model and also return the gradient
"""
function fast_value_and_gradient(p, model, Ain, Y; nsteps::Int=32, basis_caches=nothing,
    method::Symbol=:rk4, loss=squared_error_loss)
    σ = model.sigma
    d = length(model.B[1])
    nA = length(model.A)
    repA, repB = _fast_reps(model, basis_caches)
    X0 = p.W_in * Ain
    h = one(eltype(X0)) / nsteps
    field(X, t) = _f_flat(X, p, t, σ, repA, repB, d, nA)
    traj = _fixedstep_traj(X0, nsteps, h, field; method)
    Xf = traj[nsteps+1]
    E, dZ = loss_value_and_grad(loss, p.W_out * Xf, Y)

    ∇E = zero(p)
    ∇E.W_out .= dZ * Xf'
    costate = p.W_out' * dZ
    vjp_at(X, t) = ChainRulesCore.rrule(_f_flat, X, p, t, σ, repA, repB, d, nA)
    costate = _fixedstep_adjoint!(∇E, costate, traj, nsteps, h, vjp_at; method)
    ∇E.W_in .= costate * Ain'
    return E, ∇E
end

# ==== Residual block fast path ====

_block_eval(model::ResBlockFlow) =
    build_block(model.spec, model.sigma, cpu_field_ops(model.As[1], model.Bs[1]))

"""
    block_fast_value(p, model::ResBlockFlow, Ain, Y; nsteps=32, method=:rk4, loss=…) → E
"""
function block_fast_value(p, model::ResBlockFlow, Ain, Y; nsteps::Int=32,
    method::Symbol=:rk4, loss=squared_error_loss)
    be = _block_eval(model)
    X0 = p.W_in * Ain
    h = one(eltype(X0)) / nsteps
    field(X, t) = first(be(X, p, t))
    Xf = _fixedstep_traj(X0, nsteps, h, field; method)[nsteps+1]
    return loss(p.W_out * Xf, Y)
end

"""
    block_fast_value_and_gradient(p, model::ResBlockFlow, Ain, Y; nsteps=32, method=:rk4) → (E, ∇E)

Fixed-step adjoint for the depth-N block model.
"""
function block_fast_value_and_gradient(p, model::ResBlockFlow, Ain, Y; nsteps::Int=32,
    method::Symbol=:rk4, loss=squared_error_loss)
    be = _block_eval(model)
    X0 = p.W_in * Ain
    h = one(eltype(X0)) / nsteps
    field(X, t) = first(be(X, p, t))
    traj = _fixedstep_traj(X0, nsteps, h, field; method)
    Xf = traj[nsteps+1]
    E, dZ = loss_value_and_grad(loss, p.W_out * Xf, Y)

    ∇E = zero(p)
    ∇E.W_out .= dZ * Xf'
    costate = p.W_out' * dZ
    vjp_at(X, t) = (yp = be(X, p, t); (yp[1], cot -> (nothing, yp[2](cot)...)))
    costate = _fixedstep_adjoint!(∇E, costate, traj, nsteps, h, vjp_at; method)
    ∇E.W_in .= costate * Ain'
    return E, ∇E
end

"""
Wrapper to choose between adaptive and fixed steps.
"""
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

"""
Solve and return the final state x(1) for a batch
"""
function _batch_final(p, model, Ain, alg, reltol, abstol, d, nA, dt=nothing,
    repA=model.A, repB=model.B, sensealg=_default_sensealg())
    X0 = p.W_in * Ain
    rhs(X, q, t) = _f_flat(X, q, t, model.sigma, repA, repB, d, nA)
    prob = ODEProblem(rhs, X0, (zero(eltype(X0)), one(eltype(X0))), p)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, _step_opts(dt)...,
        sensealg=sensealg,
        save_everystep=false)
    return sol.u[end]
end

"""
Calculate loss over all samples in a batch
"""
_batch_loss(p, model, Ain, Y, alg, reltol, abstol, d, nA, dt=nothing,
    repA=model.A, repB=model.B, sensealg=_default_sensealg(), loss=squared_error_loss) =
    loss(p.W_out * _batch_final(p, model, Ain, alg, reltol, abstol, d, nA, dt, repA, repB, sensealg), Y)

"""
    batched_value_and_gradient(p, model, Ain, Y; solver=Tsit5(), reltol, abstol)
        → (E, g::ComponentVector)

Compute E(P) and its gradient for a whole batch.
"""
function batched_value_and_gradient(p, model::NeuralFlowODE,
    Ain::AbstractMatrix, Y::AbstractMatrix;
    solver=Tsit5(), reltol=1e-9, abstol=1e-12, dt=nothing,
    basis_caches::Union{Nothing,Tuple}=nothing, sensealg=_default_sensealg(),
    loss=squared_error_loss)
    d = length(model.B[1])
    nA = length(model.A)
    repA, repB = basis_caches === nothing ? (model.A, model.B) : basis_caches
    E, grads = Zygote.withgradient(
        q -> _batch_loss(q, model, Ain, Y, solver, reltol, abstol, d, nA, dt,
            repA, repB, sensealg, loss), p)
    g = grads[1]
    g isa ComponentVector || (g = ComponentVector(g, getaxes(p)))
    return E, g
end

"""
    batched_predict(model, inputs; solver=Tsit5(), reltol, abstol) → output_dim × N

Compute the model output W_out · x(1) for an entire batch.
"""
function batched_predict(model::NeuralFlowODE, inputs::Vector;
    solver=Tsit5(), reltol=1e-9, abstol=1e-12, dt=nothing,
    basis_caches::Union{Nothing,Tuple}=nothing)
    p = param_vector(model)
    Ain = _stack(inputs, eltype(p))
    d = length(model.B[1])
    nA = length(model.A)
    repA, repB = basis_caches === nothing ? (model.A, model.B) : basis_caches
    return p.W_out * _batch_final(p, model, Ain, solver, reltol, abstol, d, nA, dt,
        repA, repB)
end

# ==== Generic training helpers ====

stack_inputs(::NeuralFlowODE, inputs, ::Type{ET}) where {ET} = _stack(inputs, ET)

_field_reps(model::NeuralFlowODE) = (model.A, model.B)

_eval_caches(model, Ain) = map(r -> BasisCache(r, Ain), _field_reps(model))

_eval_loss(p, model::NeuralFlowODE, Ain, Y, caches, solver, reltol, abstol, dt, sensealg,
    loss=squared_error_loss) =
    _batch_loss(p, model, Ain, Y, solver, reltol, abstol,
        length(model.B[1]), length(model.A), dt, caches[1], caches[2], sensealg, loss)

_resume_into!(model, loaded) = set_params!(model, param_vector(loaded))

"""
    adjoint_value_and_gradients(model, inputs, targets; solver=Tsit5(), reltol, abstol)
        → (E, grad_A, grad_B, grad_W_in, grad_W_out)

Compute E(P) and its gradient with the adjoint method.
"""
function adjoint_value_and_gradients(model::NeuralFlowODE,
    inputs::Vector, targets::Vector;
    solver=Tsit5(), reltol=1e-9, abstol=1e-12)
    p = param_vector(model)
    Ain = _stack(inputs, eltype(p))
    Y = _stack(targets, eltype(p))
    E, g = batched_value_and_gradient(p, model, Ain, Y;
        solver=solver, reltol=reltol, abstol=abstol)
    grad_A = [collect(g.A[:, :, k]) for k in 1:length(model.A)]
    grad_B = [collect(g.B[:, k]) for k in 1:length(model.B)]
    return E, grad_A, grad_B, collect(g.W_in), collect(g.W_out)
end

"""
    adjoint_gradients(model, inputs, targets; solver=Tsit5(), reltol, abstol)
        → (grad_A, grad_B, grad_W_in, grad_W_out)

Convenience wrapper. Same as `adjoint_value_and_gradients`, drops the error E(P).
"""
function adjoint_gradients(model::NeuralFlowODE, inputs::Vector, targets::Vector;
    solver=Tsit5(), reltol=1e-9, abstol=1e-12)
    _, grad_A, grad_B, grad_W_in, grad_W_out =
        adjoint_value_and_gradients(model, inputs, targets;
            solver=solver, reltol=reltol, abstol=abstol)
    return grad_A, grad_B, grad_W_in, grad_W_out
end

"""
    train_adjoint(model, generate_batch; solver=Tsit5(), data_to_device=identity, kwargs...)
        → (model, errors)

Train model with the adjoint method.
"""
function train_adjoint(model, generate_batch;
    learning_rate=1e-3, n_epochs=100, verbose=10,
    num_samples=128, patience=250, lr_decay=0.75, min_lr=1e-8,
    solver=Tsit5(),
    optimizer=AdamFlat,
    data_to_device=identity,
    reltol=1e-9, abstol=1e-12, dt=nothing,
    eval_every::Int=1, sensealg=nothing,
    backend::Symbol=:sciml, nsteps::Int=32,
    loss=squared_error_loss,
    on_epoch::Union{Nothing,Function}=nothing,
    checkpoint_every::Int=0,
    checkpoint_path::Union{Nothing,String}=nothing,
    resume_from::Union{Nothing,String}=nothing)

    prior_errors = Float64[]
    if resume_from !== nothing
        saved = load_run(resume_from)
        _resume_into!(model, saved.best_model)
        prior_errors = collect(saved.errors)
    end
    epochs_left = max(0, n_epochs - length(prior_errors))

    p = data_to_device(param_vector(model))
    sensealg === nothing && (sensealg = _auto_sensealg(p))
    opt = optimizer(p)
    sync!() = set_params!(model, adapt(Array, p))

    eval_inputs, eval_targets = generate_batch()
    Ain_eval = data_to_device(stack_inputs(model, eval_inputs, eltype(p)))
    Y_eval = data_to_device(_stack(eval_targets, eltype(p)))

    caches = _eval_caches(model, Ain_eval)

    solver_name = nameof(typeof(solver))
    label = "Adjoint($solver_name)"

    spawn_batch() = Threads.@spawn begin
        inputs, targets = generate_batch()
        (data_to_device(stack_inputs(model, inputs, eltype(p))), data_to_device(_stack(targets, eltype(p))))
    end
    next_batch = spawn_batch()

    iters = Ref(0)
    # :fasttsit5/:fastadaptive are batched-only, we fallback to :sciml instead for nonbatched
    isfast = backend === :fastrk4 || backend === :fasteuler
    fastmethod = backend === :fasteuler ? :euler : :rk4
    gradfn(pp, Ain, Y) = isfast ?
                         fast_value_and_gradient(pp, model, Ain, Y; nsteps, basis_caches=caches, method=fastmethod, loss=loss) :
                         batched_value_and_gradient(pp, model, Ain, Y;
        solver=solver, reltol=reltol, abstol=abstol, dt=dt,
        basis_caches=caches, sensealg=sensealg, loss=loss)
    function step!(lr)
        Ain, Y = fetch(next_batch)
        next_batch = spawn_batch()
        E_train, g = gradfn(p, Ain, Y)
        apply_gradients!(opt, p, lr, g)
        iters[] += 1
        if eval_every > 0 && iters[] % eval_every == 0
            return isfast ?
                   fast_value(p, model, Ain_eval, Y_eval; nsteps, basis_caches=caches, method=fastmethod, loss=loss) :
                   _eval_loss(p, model, Ain_eval, Y_eval, caches, solver, reltol,
                abstol, dt, sensealg, loss)
        end
        return E_train
    end

    cp = nothing
    callback = nothing
    if checkpoint_path !== nothing || on_epoch !== nothing
        if checkpoint_path !== nothing
            cp = Checkpointer(checkpoint_path, checkpoint_every;
                metadata=(solver=solver_name,
                    num_samples=num_samples, n_epochs=n_epochs))
        end
        callback = (E, epoch, errors) -> begin
            if cp !== nothing
                if E < cp.best
                    sync!()
                    on_improve!(cp, E, epoch, copy_model(model))
                end
                checkpoint!(cp, epoch, vcat(prior_errors, errors))
            end
            if on_epoch !== nothing
                sync!()
                on_epoch(E, epoch, errors)
            end
        end
    end

    errors = train_loop(step!, epochs_left, num_samples;
        learning_rate, verbose, patience, lr_decay, min_lr, label=label,
        on_epoch=callback)

    sync!()
    full_errors = vcat(prior_errors, errors)
    return model, full_errors
end

function train_determinant_adjoint(matrix_size, hidden_dim, ncp;
    sigma=relu, num_samples=128, input_scale=1.0,
    learning_rate=1e-3, n_epochs=100, verbose=10,
    patience=250, lr_decay=0.75, min_lr=1e-8,
    solver=Tsit5(), RepType::Type=Spline)
    task = DeterminantTask(matrix_size; input_scale=input_scale)
    model = init_model(task, hidden_dim, ncp; sigma=sigma, RepType=RepType)
    generate_batch() = generate(task, num_samples)
    return train_adjoint(model, generate_batch; learning_rate, n_epochs, verbose, num_samples,
        patience, lr_decay, min_lr, solver)
end

end
