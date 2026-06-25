"""
Implements different ODE solvers.
All solvers must take (model, a, time_steps) and return (times, solutions).
x0=a is the initial condition
"""
module Solvers

using LinearAlgebra
using QuadGK
using DifferentialEquations
using ..LinearInterp
using ..ODEModel

export solve_euler, solve_tsit5, solve_picard

"""
    solve_euler(model, a, time_steps) → (times, solutions)

Solve the model using Euler.
"""
function solve_euler(model::AbstractFlowModel,
    a::AbstractVecOrMat{<:Real},
    time_steps::AbstractVector{<:Real})
    N = length(time_steps)
    T = promote_type(eltype(a), field_eltype(model))
    xs = Vector{typeof(T.(a))}(undef, N)
    xs[1] = T.(a)
    for k in 1:(N-1)
        Δt = time_steps[k+1] - time_steps[k]
        xs[k+1] = xs[k] .+ Δt .* f(model, xs[k], time_steps[k])
    end
    return time_steps, xs
end

"""
    solve_tsit5(model, a, tgrid; reltol=1e-9, abstol=1e-12) → (times, solutions)

Solve the model with Tsit5 (DifferentialEquations.jl).
"""
function solve_tsit5(model::AbstractFlowModel,
    a::AbstractVecOrMat{<:Real},
    tgrid::AbstractVector{<:Real};
    reltol::Float64=1e-9,
    abstol::Float64=1e-12)
    rhs!(dx, x, _, t) = (dx .= f(model, x, t))
    prob = ODEProblem(rhs!, a, (minimum(tgrid), maximum(tgrid)))
    sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol, saveat=tgrid)
    return sol.t, sol.u
end

"""
    solve_picard(model, a, time_steps; maxit=30, atol=1e-10) → (times, solutions)

Solve F(x, a, P) = x(t) - a - ∫₀ᵗ f(x(s), s) ds = 0 using Picard iteration.
"""
function solve_picard(model::AbstractFlowModel,
    a::AbstractVector{<:Real},
    time_steps::AbstractVector{<:Real};
    maxit::Int=30,
    atol::Float64=1e-10)
    xs = [copy(a) for _ in 1:length(time_steps)]

    for it in 1:maxit
        xs_new = similar(xs)
        max_err = 0.0

        for (i, t) in enumerate(time_steps)
            I, _ = quadgk(
                s -> f(model, interpolate_samples(xs, s), s),
                0.0, t; rtol=1e-10, atol=1e-12
            )
            xs_new[i] = a .+ I
            max_err = max(max_err, norm(xs_new[i] .- xs[i]))
        end

        xs .= xs_new
        if max_err < atol
            @info "Picard converged after $it iterations (max error = $max_err)"
            break
        end
        it == maxit && @warn "Picard did not reach tolerance; final max error = $max_err"
    end

    return time_steps, xs
end

end
