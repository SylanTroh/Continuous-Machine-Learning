"""
Finite-difference gradient of E(P) via FiniteDifferences.jl.
Used as reference for the other gradient methods.
Incredibly slow.
"""
module FiniteDiff

using ComponentArrays
using FiniteDifferences
using ..AbstractRepresentation
using ..ODEModel
using ..Solvers
using ..ModelUtils: param_vector, set_params!, copy_model
using ..TrainingUtils: compute_error

export fd_gradients, fd_gradient_flat, fd_loss

"""
    fd_loss(model, inputs, targets; reltol=1e-12, abstol=1e-14) → Float64

Total squared-error loss via Tsit5
"""
fd_loss(model::AbstractFlowModel, inputs, targets;
    reltol::Float64=1e-12, abstol::Float64=1e-14) =
    compute_error(model, inputs, targets;
        solver=(m, a, ts) -> solve_tsit5(m, a, ts; reltol=reltol, abstol=abstol))

"""
    fd_gradient_flat(model, inputs, targets; reltol, abstol, fdm) → ComponentVector

Finite-difference gradient of E(P) as a flat parameter vector.
"""
function fd_gradient_flat(model::AbstractFlowModel, inputs, targets;
    reltol::Float64=1e-12, abstol::Float64=1e-14, fdm=central_fdm(5, 1))
    work = copy_model(model)
    p₀ = param_vector(model)
    ax = getaxes(p₀)
    function loss(pvec)
        set_params!(work, ComponentVector(pvec, ax))
        return fd_loss(work, inputs, targets; reltol=reltol, abstol=abstol)
    end
    return ComponentVector(FiniteDifferences.grad(fdm, loss, Vector(p₀))[1], ax)
end

"""
    fd_gradients(model, inputs, targets; reltol=1e-12, abstol=1e-14, fdm=central_fdm(5, 1))
        → (grad_A, grad_B, grad_W_in, grad_W_out)

Finite-difference parameter gradient.
"""
function fd_gradients(model::NeuralFlowODE, inputs, targets;
    reltol::Float64=1e-12, abstol::Float64=1e-14, fdm=central_fdm(5, 1))
    g = fd_gradient_flat(model, inputs, targets; reltol, abstol, fdm)
    grad_A = [g.A[:, :, k] for k in eachindex(vals(model.A))]
    grad_B = [g.B[:, k] for k in eachindex(vals(model.B))]
    return grad_A, grad_B, collect(g.W_in), collect(g.W_out)
end

end
