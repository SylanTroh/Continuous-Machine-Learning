#!/usr/bin/env julia
# Usage:  OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/diag_ramp.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "setup", "gpu_env.jl"))
using NeuralFlow
using DifferentialEquations: Tsit5
using Random
using Printf
using Statistics: mean, std, median
using LinearAlgebra: BLAS
using CUDA, Adapt
include(joinpath(@__DIR__, "cuda_utils.jl"))
include(joinpath(@__DIR__, "batched_sweep.jl"))

BLAS.set_num_threads(1)

const N = 4
const NCP = 8
const VARIANT = :plain
const DIMS = [64]
const NSEEDS = 10
const SEEDS = collect(1:NSEEDS)
const RHO_FIX = 0.0

const ARMS = [
    (:diag_ramp, [(0.0, 0.0), (0.8, 1.0), (1.0, 1.0)]),
]

const EXPERIMENT = get(ENV, "SR_EXP", "diag_ramp")
const USE_GPU = get(ENV, "SR_GPU", "auto")
const BACKEND = :fastrk4

const RELTOL = 1e-3
const ABSTOL = 1e-6
const PATIENCE = 500
const LR_DECAY = 0.75
const MIN_LR = 1e-6
const LR = 1e-2
const NEPOCHS = 20000

const PRECISION = Float32
const SIGMA = relu
const REP = Spline
const SOLVER = Tsit5()
const SEED_BATCH = 100
const NSAMP = 64
const N_FRESH = 1024

const THRESHOLD = 0.10
const TRACK_N = 256
const LOG_EVERY = 250
const CHECKPOINT_EVERY = 4 * LOG_EVERY
const PLATEAU_EVALS = 0
const MIN_IMPROVE = 0.01

function _variant_base(variant::Symbol, n::Int, seed::Int)
    variant === :cofactor && return CofactorDeterminantTask(n)
    variant === :nested && return NestedDeterminantTask(n)
    variant === :random && return RandomMinorDeterminantTask(n; m=2, seed=seed)
    variant === :plain && return DeterminantTask(n)
    error("unknown variant $variant")
end

main_only(m) = NeuralFlowODE(m.A, m.B, m.W_in, m.W_out[1:1, :], m.sigma)

function _rho_schedule(p::Real, knots)
    p <= knots[1][1] && return knots[1][2]
    for i in 2:length(knots)
        if p <= knots[i][1]
            (p0, v0), (p1, v1) = knots[i-1], knots[i]
            w = (p - p0) / (p1 - p0)
            return v0 + w * (v1 - v0)
        end
    end
    return knots[end][2]
end

function diag_sampler(base, n::Int, num_samples::Int, knots, n_epochs::Int; rng)
    calls = Ref(0)
    return () -> begin
        calls[] += 1
        p = min(1.0, (calls[] - 1) / max(1, n_epochs))
        ρ = _rho_schedule(p, knots)
        inputs = Matrix{Float64}[]
        targets = Vector{Float64}[]
        for _ in 1:num_samples
            M = base.input_scale .* randn(rng, n, n)
            @inbounds for j in 1:n, i in 1:n
                i != j && (M[i, j] *= ρ)
            end
            push!(inputs, M)
            push!(targets, compute_target(base, M))
        end
        return inputs, targets
    end
end

function _bias_matrix(M::AbstractMatrix, ρ::Real, n::Int)
    B = copy(M)
    @inbounds for j in 1:n, i in 1:n
        i != j && (B[i, j] *= ρ)
    end
    return B
end

function arm_cell(arm::Symbol, knots, h::Int;
    variant::Symbol=VARIANT, n::Int=N, ncp::Int=NCP,
    seeds::AbstractVector{<:Integer}=SEEDS, seed_batch::Int=SEED_BATCH,
    n_epochs::Int=NEPOCHS, num_samples::Int=NSAMP,
    log_every::Int=LOG_EVERY, plateau_evals::Int=PLATEAU_EVALS, min_improve::Real=MIN_IMPROVE,
    threshold::Float64=THRESHOLD, sigma=SIGMA, RepType::Type=REP, T::Type=PRECISION,
    learning_rate=LR, solver=SOLVER, reltol=RELTOL, abstol=ABSTOL, dt=nothing,
    patience::Int=PATIENCE, lr_decay=LR_DECAY, min_lr=MIN_LR,
    backend::Symbol=BACKEND, nsteps::Int=ncp, n_fresh::Int=N_FRESH,
    data_to_device=identity, experiment::String=EXPERIMENT, lg=nothing)

    name = "$(variant)_$(arm)_n$(n)_h$(h)"
    path = cell_path(experiment, name)
    isfile(path) && !resume_enabled() && return load_run(path).result

    M_seeds = length(seeds)
    logln(lg, "Cell $name: $M_seeds seeds in batches of $seed_batch  (ρ-schedule=$arm)")

    ρT = _rho_schedule(1.0, knots)
    indist_d = Dict{Int,Float64}()
    fresh_d = Dict{Int,Float64}()
    conv_d = Dict{Int,Bool}()
    e2τ_d = Dict{Int,Union{Int,Nothing}}()
    track_d = Dict{Int,Any}()
    rep = Ref{Any}(nothing)

    warm = load_models(experiment, name)
    warm !== nothing &&
        logln(lg, "  warm-starting $(count(!isnothing, warm.weights))/$M_seeds seeds")

    process = ctx -> begin
        batch = ctx.batch
        models = ctx.models
        bases = [_variant_base(variant, n, s) for s in batch]
        gens = [diag_sampler(bases[m], n, num_samples, knots, n_epochs;
                    rng=Xoshiro(batch[m] + DATA_OFFSET)) for m in eachindex(batch)]
        heldout = [generate(bases[m], n_fresh; rng=Xoshiro(batch[m] + EVAL_OFFSET))
                   for m in eachindex(batch)]
        Ys = [reduce(hcat, ho[2]) for ho in heldout]
        baselines = [target_baseline(Y) for Y in Ys]

        indist = [let Ms = [_bias_matrix(M, ρT, n) for M in heldout[m][1]]
                      (Ms, [compute_target(bases[m], M) for M in Ms])
                  end for m in eachindex(batch)]
        Yi = [reduce(hcat, id[2]) for id in indist]
        base_in = [target_baseline(Y) for Y in Yi]

        track_n = min(TRACK_N, n_fresh)
        tr_in = [indist[m][1][1:track_n] for m in eachindex(batch)]
        tr_Y = [Yi[m][:, 1:track_n] for m in eachindex(batch)]
        tr_base = [target_baseline(tr_Y[m]) for m in eachindex(batch)]
        trackers = [ConvergenceTracker(threshold; every=log_every, plateau_evals, min_improve)
                    for _ in batch]
        cbs = [tracker_callback(trackers[m],
            () -> normalized_eval(models[m], tr_in[m], tr_Y[m], tr_base[m];
                solver, reltol, abstol, dt);
            metric=first, lg, label="$(name)_s$(batch[m])") for m in eachindex(batch)]
        track = (E, epoch, errors) -> (foreach(cb -> cb(E, epoch, errors), cbs); nothing)
        on_epoch = combine_callbacks(track, ctx.save_cb)

        train_batched(models, gens; solver, n_epochs, num_samples,
            learning_rate=ctx.resume_lr, patience, lr_decay, min_lr, data_to_device,
            reltol, abstol, dt, eval_every=0, sync_every=log_every, verbose=0,
            on_loss=ctx.record_lr, on_epoch, backend, nsteps)

        if CUDA.functional()
            CUDA.synchronize()
            GC.gc(); GC.gc()
            CUDA.reclaim()
        end

        mains = [main_only(models[m]) for m in eachindex(batch)]
        indist_gens = [let inp = indist[m][1], tgt = [[t[1]] for t in indist[m][2]]
                           () -> (inp, tgt)
                       end for m in eachindex(batch)]
        full_gens = [let inp = heldout[m][1], tgt = [[t[1]] for t in heldout[m][2]]
                         () -> (inp, tgt)
                     end for m in eachindex(batch)]
        indist_ek = batched_heldout_errors(mains, indist_gens;
            backend, nsteps, solver, reltol, abstol, dt, data_to_device)
        fresh_ek = batched_heldout_errors(mains, full_gens;
            backend, nsteps, solver, reltol, abstol, dt, data_to_device)

        for m in eachindex(batch)
            s = batch[m]
            indist_d[s] = indist_ek[m] / base_in[m][1]
            fresh_d[s] = fresh_ek[m] / baselines[m][1]
            conv_d[s] = converged(trackers[m])
            e2τ_d[s] = trackers[m].epoch_converged
            track_d[s] = track_matrix(trackers[m], length(baselines[m]))
            if rep[] === nothing
                rep[] = (track_epochs=trackers[m].epochs,
                    track=track_matrix(trackers[m], length(baselines[m])),
                    n_params=n_params(models[m]))
            end
        end
        CUDA.functional() && (GC.gc(); CUDA.reclaim())
    end

    t0 = time()
    run_seed_sweep(seeds; experiment, tag=name, seed_batch, per_seed_schedule=false,
        base_lr=learning_rate, checkpoint_every=CHECKPOINT_EVERY, lg, label=name,
        init_model=(s -> init_model(_variant_base(variant, n, s), h, ncp;
            sigma, RepType, init_scale=0.3, T, rng=Xoshiro(s))),
        process,
        init_models=(warm === nothing ? nothing : warm.weights),
        init_lr=(warm === nothing ? nothing : warm.lr))
    wall = time() - t0

    indist_s = [indist_d[s] for s in seeds]
    fresh_s = [fresh_d[s] for s in seeds]
    conv_s = [conv_d[s] for s in seeds]
    e2τ_s = Union{Int,Nothing}[e2τ_d[s] for s in seeds]
    track_s = [get(track_d, s, nothing) for s in seeds]
    e2τ_hit = [e for e in e2τ_s if e !== nothing]
    repr = rep[]
    det_EVar = mean(indist_s)
    det_EVar_full = mean(fresh_s)
    conv_frac = mean(conv_s)
    logln(lg, @sprintf("Done %s (ρT=%.2f): in-dist E/Var=%.4f±%.4f  full-random E/Var=%.4f  conv=%d/%d wall=%.0fs",
        name, ρT, det_EVar, std(indist_s; corrected=false), det_EVar_full,
        count(conv_s), M_seeds, wall))

    config = (sigma=Symbol(sigma), rep=string(nameof(RepType)),
        solver=string(nameof(typeof(solver))), T=string(T),
        backend=string(backend), nsteps=nsteps, learning_rate=learning_rate,
        num_samples=num_samples, seeds=collect(seeds), seed_batch=seed_batch,
        rho_fix=RHO_FIX, knots=knots)
    result = (variant=variant, arm=arm, tag=string(arm), n=n, h=h, ncp=ncp,
        n_params=repr.n_params, threshold=threshold, n_epochs=n_epochs, rho_fix=RHO_FIX,
        knots=knots, terminal_rho=ρT,
        det_EVar=det_EVar, det_EVar_std=std(indist_s; corrected=false),
        det_EVar_full=det_EVar_full, det_EVar_full_std=std(fresh_s; corrected=false),
        converged=conv_frac >= 0.5, converged_frac=conv_frac,
        epochs_to_τ=isempty(e2τ_hit) ? nothing : round(Int, median(e2τ_hit)),
        n_seeds=M_seeds, seeds=collect(seeds),
        per_seed=(det_EVar=indist_s, det_EVar_full=fresh_s, converged=conv_s,
            epochs_to_τ=e2τ_s, track=track_s),
        track_epochs=repr.track_epochs, track=repr.track, wall=wall, config=config)
    save_run(path; result=result)
    return result
end

function diag_ramp(; variant::Symbol=VARIANT, dims=DIMS, n::Int=N, arms=ARMS,
    experiment::String=EXPERIMENT, lg=nothing, kwargs...)
    cells = [() -> arm_cell(name, knots, h; variant, n, experiment, lg, kwargs...)
             for h in dims for (name, knots) in arms]
    results = run_parallel(cells; max_concurrent=1)
    logln(lg, "Summary")
    for r in results
        logln(lg, @sprintf("  %-11s n=%d h=%-4d ρT=%.2f  in-dist=%.4f±%.4f  full=%.4f  %s",
            r.tag, r.n, r.h, r.terminal_rho, r.det_EVar, r.det_EVar_std, r.det_EVar_full,
            r.converged ? "Converged" : "not converged"))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    lg = TeeLog(EXPERIMENT, "sweep")
    diag_ramp(lg=lg; data_to_device=gpu_device(USE_GPU))
    close(lg)
end
