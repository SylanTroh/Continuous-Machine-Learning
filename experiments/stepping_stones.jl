#!/usr/bin/env julia
# Usage:  OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/stepping_stones.jl

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
const VARIANTS = [:nested, :cofactor]
const LEVELS = collect(1:N)
const HIDDEN = 64
const NCP = 4
const NSEEDS = 100
const SEEDS = collect(1:NSEEDS)

const EXPERIMENT = get(ENV, "SS_EXP", "stepping_stones")
const USE_GPU = get(ENV, "SS_GPU", "auto")
const BACKEND = :fastrk4

const RELTOL = 1e-3
const ABSTOL = 1e-6
const PATIENCE = 500
const LR_DECAY = 0.75
const MIN_LR = 1e-6
const LR = 3e-3
const NEPOCHS = 40000

const PRECISION = Float32
const SIGMA = tanh
const REP = ChebPoly
const SOLVER = Tsit5()
const SEED_BATCH = 100
const NSAMP = 256
const N_FRESH = 1024

const THRESHOLD = 0.0
const TRACK_N = 256
const LOG_EVERY = 250
const CHECKPOINT_EVERY = 4 * LOG_EVERY
const PLATEAU_EVALS = 6
const MIN_IMPROVE = 0.01

function _arm_variant(variant::Symbol, n::Int, levels::Int)
    variant === :nested && return NestedDeterminantTask(n; levels)
    variant === :cofactor && return CofactorDeterminantTask(n; levels)
    error("unknown variant $variant")
end


main_only(m) = NeuralFlowODE(m.A, m.B, m.W_in, m.W_out[1:1, :], m.sigma)

function arm_cell(variant::Symbol, levels::Int;
    n::Int=N, hidden_dim::Int=HIDDEN, ncp::Int=NCP,
    seeds::AbstractVector{<:Integer}=SEEDS, seed_batch::Int=SEED_BATCH,
    n_epochs::Int=NEPOCHS, num_samples::Int=NSAMP,
    log_every::Int=LOG_EVERY, plateau_evals::Int=PLATEAU_EVALS, min_improve::Real=MIN_IMPROVE,
    threshold::Float64=THRESHOLD, sigma=SIGMA, RepType::Type=REP, T::Type=PRECISION,
    learning_rate=LR, solver=SOLVER, reltol=RELTOL, abstol=ABSTOL, dt=nothing,
    patience::Int=PATIENCE, lr_decay=LR_DECAY, min_lr=MIN_LR,
    backend::Symbol=BACKEND, nsteps::Int=ncp, n_fresh::Int=N_FRESH,
    data_to_device=identity, experiment::String=EXPERIMENT, lg=nothing)

    tag = "$(variant)_L$(levels)"
    name = "$(tag)_n$(n)"
    path = cell_path(experiment, name)
    isfile(path) && !resume_enabled() && return load_run(path).result

    tk = _arm_variant(variant, n, levels)
    M = length(seeds)
    logln(lg, "Cell $name: $M seeds in batches of $seed_batch  ($(variant), depth $(levels), $(output_dim(tk)) outputs)")

    fresh_d = Dict{Int,Float64}()
    conv_d = Dict{Int,Bool}()
    e2τ_d = Dict{Int,Union{Int,Nothing}}()
    track_d = Dict{Int,Any}()
    rep = Ref{Any}(nothing)

    warm = load_models(experiment, name)
    warm !== nothing &&
        logln(lg, "  warm-starting $(count(!isnothing, warm.weights))/$M seeds")

    process = ctx -> begin
        batch = ctx.batch
        models = ctx.models
        gens = [let r = Xoshiro(s + DATA_OFFSET)
                    () -> generate(tk, num_samples; rng=r)
                end for s in batch]
        heldout = [generate(tk, n_fresh; rng=Xoshiro(s + EVAL_OFFSET)) for s in batch]
        Ys = [reduce(hcat, ho[2]) for ho in heldout]
        baselines = [target_baseline(Y) for Y in Ys]

        track_n = min(TRACK_N, n_fresh)
        tr_in = [heldout[m][1][1:track_n] for m in eachindex(batch)]
        tr_Y = [Ys[m][:, 1:track_n] for m in eachindex(batch)]
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
        main_gens = [let inp = heldout[m][1], tgt = [[t[1]] for t in heldout[m][2]]
                         () -> (inp, tgt)
                     end for m in eachindex(batch)]
        fresh_ek = batched_heldout_errors(mains, main_gens;
            backend, nsteps, solver, reltol, abstol, dt, data_to_device)

        for m in eachindex(batch)
            s = batch[m]
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
        init_model=(s -> init_model(tk, hidden_dim, ncp; sigma, RepType, init_scale=0.3, T,
            rng=Xoshiro(s))),
        process,
        init_models=(warm === nothing ? nothing : warm.weights),
        init_lr=(warm === nothing ? nothing : warm.lr))
    wall = time() - t0

    fresh_s = [fresh_d[s] for s in seeds]
    conv_s = [conv_d[s] for s in seeds]
    e2τ_s = Union{Int,Nothing}[e2τ_d[s] for s in seeds]
    track_s = [get(track_d, s, nothing) for s in seeds]
    e2τ_hit = [e for e in e2τ_s if e !== nothing]
    repr = rep[]
    det_EVar = mean(fresh_s)
    conv_frac = mean(conv_s)
    logln(lg, @sprintf("Done %s: det E/Var=%.4f±%.4f conv=%d/%d wall=%.0fs",
        name, det_EVar, std(fresh_s; corrected=false), count(conv_s), M, wall))

    config = (sigma=Symbol(sigma), rep=string(nameof(RepType)),
        solver=string(nameof(typeof(solver))), T=string(T),
        backend=string(backend), nsteps=nsteps, learning_rate=learning_rate,
        num_samples=num_samples, seeds=collect(seeds), seed_batch=seed_batch)
    result = (variant=variant, levels=levels, tag=tag, n=n, n_aux=output_dim(tk) - 1,
        output_dim=output_dim(tk), hidden_dim=hidden_dim, ncp=ncp, n_params=repr.n_params,
        threshold=threshold, n_epochs=n_epochs,
        det_EVar=det_EVar, det_EVar_std=std(fresh_s; corrected=false),
        main_EVar=det_EVar,
        converged=conv_frac >= 0.5, converged_frac=conv_frac,
        epochs_to_τ=isempty(e2τ_hit) ? nothing : round(Int, median(e2τ_hit)),
        n_seeds=M, seeds=collect(seeds),
        per_seed=(det_EVar=fresh_s, converged=conv_s, epochs_to_τ=e2τ_s, track=track_s),
        track_epochs=repr.track_epochs, track=repr.track, wall=wall, config=config)
    save_run(path; result=result)
    return result
end

function stepping_stones(; n::Int=N, variants=VARIANTS, levels=LEVELS,
    experiment::String=EXPERIMENT, lg=nothing, kwargs...)
    cells = Function[]
    for var in variants, L in levels
        L == 1 && var != first(variants) && continue
        push!(cells, () -> arm_cell(var, L; n, experiment, lg, kwargs...))
    end
    results = run_parallel(cells; max_concurrent=1)
    logln(lg, "Summary (det-component E/Var by variant × depth)")
    for r in results
        logln(lg, @sprintf("  %-14s depth=%d (%d aux)  det E/Var=%.4f±%.4f  conv=%.0f%%",
            r.tag, r.levels, r.n_aux, r.det_EVar, r.det_EVar_std, 100 * r.converged_frac))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    lg = TeeLog(EXPERIMENT, "run")
    stepping_stones(lg=lg; data_to_device=gpu_device(USE_GPU))
    close(lg)
end
