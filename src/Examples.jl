"""
Examples:
- `example_random()` compare the solvers on a random model
- `example_train_determinant()` train a model to predict matrix determinants
"""
module Examples

using LinearAlgebra
using Random
using Printf
using Plots
using ..AbstractRepresentation
using ..SplineRepresentation
using ..ChebyshevRepresentation
using ..ActivationFunctions
using ..ODEModel
using ..Solvers
using ..ModelInit
using ..Tasks
using ..Adjoint
using ..MNISTData

export example_random, example_train_determinant

"""
    example_random(; kwargs...)

Randomly initialize a NeuralFlowODE and compare solvers (Euler, Tsit5, Picard).
"""
function example_random(; dim::Int=3,
    ncp::Int=7,
    t_end::Float64=1.0,
    n_time::Int=300,
    sigma=relu,
    picard_maxit::Int=30,
    picard_atol::Float64=1e-12,
    RepType::Type=ChebPoly)
    A_repr = random_like(RepType, ncp, (dim, dim))
    B_repr = random_like(RepType, ncp, (dim,))
    model = NeuralFlowODE(A_repr, B_repr; sigma=sigma)
    a = randn(dim)
    tgrid = collect(range(0.0, stop=t_end, length=n_time))

    @info "Computing Picard solution..."
    t_pic, xs_picard = solve_picard(model, a, tgrid; maxit=picard_maxit, atol=picard_atol)
    @info "Computing Tsit5 solution..."
    t_tsit, xs_tsit = solve_tsit5(model, a, tgrid)
    @info "Computing Euler solution..."
    t_eul, xs_euler = solve_euler(model, a, tgrid)

    err_tsit = [norm(xs_tsit[i] .- xs_picard[i]) for i in eachindex(tgrid)]
    err_euler = [norm(xs_euler[i] .- xs_picard[i]) for i in eachindex(tgrid)]
    L2_tsit = sqrt(sum(err_tsit .^ 2) * (t_end / (n_time - 1)))
    L2_euler = sqrt(sum(err_euler .^ 2) * (t_end / (n_time - 1)))

    @info "Error summary" L2_tsit = round(L2_tsit, sigdigits=6) max_tsit = round(maximum(err_tsit), sigdigits=6) L2_euler = round(L2_euler, sigdigits=6) max_euler = round(maximum(err_euler), sigdigits=6)

    p1 = plot(tgrid, getindex.(xs_picard, 1), label="Picard", lw=2)
    plot!(p1, tgrid, getindex.(xs_tsit, 1), ls=:dash, label="Tsit5", lw=2)
    plot!(p1, tgrid, getindex.(xs_euler, 1), ls=:dot, label="Euler", lw=2)
    plot!(p1; title="Solver comparison", xlabel="t", ylabel="x_1(t)")
    savefig(p1, joinpath(@__DIR__, "..", "Figures", "solver_comparison.png"))

    return (t=tgrid, picard=xs_picard, tsit5=xs_tsit, euler=xs_euler,
        err_tsit=err_tsit, err_euler=err_euler, model=model, a=a)
end

"""
    example_train_determinant(; kwargs...)

Train a NeuralFlowODE to predict matrix determinants. Plots training error E(P).
"""
function example_train_determinant(; matrix_size=2, hidden_dim=32, ncp=4,
    sigma=relu, kwargs...)
    println("Training Started")

    model = nothing
    errors = Float64[]

    try
        model, errors = train_determinant_adjoint(
            matrix_size, hidden_dim, ncp;
            sigma=sigma, num_samples=1024, verbose=50,
            n_epochs=10000, kwargs...
        )
    catch e
        isa(e, InterruptException) || rethrow()
        @info "Training interrupted."
    end

    if !isempty(errors)
        println("Final error: $(round(errors[end], digits=6))")
        p = plot(1:length(errors), errors;
            label="Training Error", lw=2,
            xlabel="Epoch", ylabel="E(P)", yscale=:log10,
            title="Determinant Prediction")
        savefig(p, joinpath(@__DIR__, "..", "Figures", "training_loss.png"))
    end

    return model, errors
end

end
