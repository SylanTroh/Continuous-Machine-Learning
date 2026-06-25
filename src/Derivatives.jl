"""
Computes the gradient ∇E(P) symbolically.
Differentiating F(x,a,P) = 0 gives DₚF = D₁F·Dₚx + D₂F = 0.
To find v = Dₚx·(dA, dB) we solve the variational equation dv/dt = D₁f·v + D₂f·(dA, dB) forward in time.
Slow, but I use it to check the adjoint method.
"""
module Derivatives

using LinearAlgebra
using ComponentArrays
using DifferentialEquations
using ForwardDiff
using ..AbstractRepresentation
using ..ActivationFunctions
using ..ODEModel
using ..Solvers
using ..ModelUtils: param_vector

export D₁f, D₂f
export solve_DPx
export forward_gradients, forward_gradients_tsit5

"""
    D₁f(model, x, t) → Matrix

Partial of f in its first argument:
D₁f(x, t; P) = diag(σ′(A(t)·x + B(t)))·A(t).
"""
function D₁f(model::NeuralFlowODE, x::AbstractVector, t::Real)
    pre_act = model.A(t) * x .+ model.B(t)
    return Diagonal(activation_derivative(model.sigma, pre_act)) * model.A(t)
end

"""
    D₁f(model::AbstractFlowModel, x, t) → Matrix

Fallback for a flow model with no closed form: state Jacobian by forward-mode AD.
"""
D₁f(model::AbstractFlowModel, x::AbstractVector, t::Real) =
    ForwardDiff.jacobian(ξ -> f(model, ξ, t), x)

"""
    D₂f(model, x, t, dA, dB) → Vector

Partial of f in its second argument:
D₂f(x, t; P)·(dA, dB) = diag(σ′(A(t)·x + B(t)))·(dA·x + dB).
"""
function D₂f(model::NeuralFlowODE, x::AbstractVector, t::Real,
    dA::AbstractMatrix, dB::AbstractVector)
    pre_act = model.A(t) * x .+ model.B(t)
    return activation_derivative(model.sigma, pre_act) .* (dA * x .+ dB)
end

"""
    solve_DPx(model, x_traj, time_steps, dA, dB) → Vector{Vector}

Directional derivative of the solution w.r.t. P along (dA, dB):
dv/dt = D₁f·v + D₂f·(dA, dB),  v(0) = 0, solved with Euler over `x_traj`.
Used to validate forward_gradients.
"""
function solve_DPx(model::NeuralFlowODE,
    x_traj::Vector{<:AbstractVector},
    time_steps::AbstractVector{<:Real},
    dA::AbstractFunctionRepresentation,
    dB::AbstractFunctionRepresentation)
    d = length(x_traj[1])
    N = length(time_steps)
    v = Vector{Vector{Float64}}(undef, N)
    v[1] = zeros(d)
    for k in 1:(N-1)
        Δt = time_steps[k+1] - time_steps[k]
        t = time_steps[k]
        x = x_traj[k]
        v[k+1] = v[k] .+ Δt .* (D₁f(model, x, t) * v[k] .+ D₂f(model, x, t, dA(t), dB(t)))
    end
    return v
end

# Layout for the flattened gradient (A, B, W_in). W_out doesn't affect the flow, so
# its gradient is computed separately.
function _param_layout(model::NeuralFlowODE, input_dim::Int)
    proto = param_vector(model)[(:A, :B, :W_in)]
    return (; ax=getaxes(proto), d=length(model.B[1]), m=input_dim,
        nA=length(model.A), nB=length(model.B),
        lA=length(proto.A), lB=length(proto.B), P=length(proto))
end

# Derivative of x(0) w.r.t. the parameters. A, B don't affect x(0) so their columns
# are zero; for W_in, x(0) = W_in·inp_vec.
function _init_DPx0(layout, inp_vec)
    DPx0 = zeros(layout.d, layout.P)
    W_cols = reshape(view(DPx0, :, layout.lA+layout.lB+1:layout.P), layout.d, layout.d, layout.m)
    for j in 1:layout.m, i in 1:layout.d
        W_cols[i, i, j] = inp_vec[j]
    end
    return DPx0
end

# Derivative of f w.r.t. the parameters. A[k][i,j], B[k][i] affect only f's ith
# component. W_in doesn't appear in f, so it stays zero.
function _compute_DPf!(DPf, model::NeuralFlowODE, x, t, σ′, layout)
    (; d, nA, nB, lA, lB) = layout
    fill!(DPf, 0.0)
    A_cols = reshape(view(DPf, :, 1:lA), d, d, d, nA)
    B_cols = reshape(view(DPf, :, lA+1:lA+lB), d, d, nB)
    wA = basis_weights(model.A, t)
    wB = basis_weights(model.B, t)
    for k in 1:nA, j in 1:d, i in 1:d
        @inbounds A_cols[i, i, j, k] = wA[k] * σ′[i] * x[j]
    end
    for k in 1:nB, i in 1:d
        @inbounds B_cols[i, i, k] = wB[k] * σ′[i]
    end
    return DPf
end

function _unpack_gradients(grad_flat, layout)
    g = ComponentVector(grad_flat, layout.ax)
    grad_A = [g.A[:, :, k] for k in 1:layout.nA]
    grad_B = [g.B[:, k] for k in 1:layout.nB]
    return grad_A, grad_B, collect(g.W_in)
end

"""
    forward_gradients(model, inputs, targets, time_steps; solver=solve_euler)
        → (grad_A, grad_B, grad_W_in, grad_W_out)

Compute the gradient of E(P) via Euler.
Solves the variational equation dV/dt = D₁f·V + DPf(t), where V = DPx
"""
function forward_gradients(model::NeuralFlowODE,
    inputs::Vector,
    targets::Vector,
    time_steps::AbstractVector{<:Real};
    solver=solve_euler)
    input_dim = size(model.W_in, 2)
    layout = _param_layout(model, input_dim)

    grad_flat = zeros(layout.P)
    grad_W_out = zeros(size(model.W_out))

    DPf = zeros(layout.d, layout.P)

    for n in eachindex(inputs)
        inp_vec = vec(inputs[n])
        a = model.W_in * inp_vec
        _, x_traj = solver(model, a, time_steps)
        x_end = x_traj[end]
        model_error = model.W_out * x_end .- targets[n]
        N = length(time_steps)

        DPx = _init_DPx0(layout, inp_vec)
        for step in 1:(N-1)
            Δt = time_steps[step+1] - time_steps[step]
            t = time_steps[step]
            x = x_traj[step]
            σ′ = activation_derivative(model.sigma, model.A(t) * x .+ model.B(t))

            _compute_DPf!(DPf, model, x, t, σ′, layout)
            DPx .= DPx .+ Δt .* (D₁f(model, x, t) * DPx .+ DPf)
        end

        # ∇E(P)⋅H = 2·model_errorᵀ·W_out·DPx·H
        # transposed below so the gradient comes out as a column vector
        grad_flat .+= DPx' * (2.0 .* (model.W_out' * model_error))
        grad_W_out .+= 2.0 .* model_error * x_end'
    end

    grad_A, grad_B, grad_W_in = _unpack_gradients(grad_flat, layout)
    return grad_A, grad_B, grad_W_in, grad_W_out
end

"""
    forward_gradients_tsit5(model, inputs, targets, time_steps; reltol, abstol)
        → (grad_A, grad_B, grad_W_in, grad_W_out)

Compute the gradient of E(P) via Tsit5. Higher-order analogue of `forward_gradients`.
"""
function forward_gradients_tsit5(model::NeuralFlowODE,
    inputs::Vector,
    targets::Vector,
    # time steps unused (Tsit5 picks its own); kept so all gradient methods share a call shape
    ::AbstractVector{<:Real}=Float64[];
    reltol::Float64=1e-9, abstol::Float64=1e-12)
    input_dim = size(model.W_in, 2)
    layout = _param_layout(model, input_dim)

    grad_flat = zeros(layout.P)
    grad_W_out = zeros(size(model.W_out))

    for n in eachindex(inputs)
        inp_vec = vec(inputs[n])
        a = model.W_in * inp_vec

        forward_rhs!(dx, x, _, t) = (dx .= f(model, x, t))
        fprob = ODEProblem(forward_rhs!, a, (0.0, 1.0))
        fsol = solve(fprob, Tsit5(); reltol=reltol, abstol=abstol, dense=true)
        x_end = fsol(1.0)

        model_error = model.W_out * x_end .- targets[n]

        DPx0 = _init_DPx0(layout, inp_vec)
        DPf = zeros(layout.d, layout.P)

        function variational_rhs!(dDPx, DPx, _, t)
            x = fsol(t)
            σ′ = activation_derivative(model.sigma, model.A(t) * x .+ model.B(t))
            _compute_DPf!(DPf, model, x, t, σ′, layout)
            mul!(dDPx, D₁f(model, x, t), DPx)
            dDPx .+= DPf
            return nothing
        end

        sprob = ODEProblem(variational_rhs!, DPx0, (0.0, 1.0))
        ssol = solve(sprob, Tsit5(); reltol=reltol, abstol=abstol, save_everystep=false)
        DPx_end = ssol.u[end]

        # ∇E(P)⋅H = 2·model_errorᵀ·W_out·DPx·H
        # transposed below so the gradient comes out as a column vector
        grad_flat .+= DPx_end' * (2.0 .* (model.W_out' * model_error))
        grad_W_out .+= 2.0 .* model_error * x_end'
    end

    grad_A, grad_B, grad_W_in = _unpack_gradients(grad_flat, layout)
    return grad_A, grad_B, grad_W_in, grad_W_out
end

end
