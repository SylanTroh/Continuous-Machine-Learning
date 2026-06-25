#!/usr/bin/env julia
# Usage:  OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/phase_diagram.jl
#
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

const NSEEDS = 100
const SEEDS = collect(1:NSEEDS)
const TASKS = AbstractTask[DeterminantTask(3), DeterminantTask(4)]
const NS = [256]
const HS = [8, 16, 32, 64]
const NCPS = [4, 8, 16, 32]

const EXPERIMENT = get(ENV, "PD_EXP", "phase_diagram")
const USE_GPU = get(ENV, "PD_GPU", "auto")
const BACKEND = :fastrk4

const RELTOL = 1e-3
const ABSTOL = 1e-6
const PATIENCE = 500
const LR_DECAY = 0.75
const MIN_LR = 1e-6
const LR = 1e-3
const NEPOCHS = 5000

const PRECISION = Float32
const SIGMA = tanh
const REP = ChebPoly
const SOLVER = Tsit5()
const SEED_BATCH = 100
const NSAMP_CAP = 256
const N_FRESH = 1024

const LOG_EVERY = 250
const CHECKPOINT_EVERY = 4 * LOG_EVERY
const PLATEAU_EVALS = 6
const MIN_IMPROVE = 0.01

function phase_cell(task; N::Int, hidden_dim::Int, ncp::Int,
    seeds::AbstractVector{<:Integer}=SEEDS, seed_batch::Int=SEED_BATCH,
    n_epochs::Int=NEPOCHS, num_samples::Int=min(N, NSAMP_CAP),
    log_every::Int=LOG_EVERY, plateau_evals::Int=PLATEAU_EVALS, min_improve::Real=MIN_IMPROVE,
    sigma=SIGMA, RepType::Type=REP, T::Type=PRECISION,
    learning_rate=LR, solver=SOLVER, reltol=RELTOL, abstol=ABSTOL, dt=nothing,
    patience::Int=PATIENCE, lr_decay=LR_DECAY, min_lr=MIN_LR,
    backend::Symbol=BACKEND, nsteps::Int=ncp,
    n_fresh::Int=N_FRESH,
    data_to_device=identity, suffix::String="",
    experiment::String=EXPERIMENT, lg=nothing)

    name = "$(task_name(task))_N$(N)_h$(hidden_dim)_ncp$(ncp)$(suffix)"
    path = cell_path(experiment, name)
    isfile(path) && !resume_enabled() && return load_run(path).result

    M = length(seeds)
    logln(lg, "Cell $name: $M seeds in batches of $seed_batch")

    in_d = Dict{Int,Float64}()
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
                    () -> generate(task, num_samples; rng=r)
                end for s in batch]

        mon_sets = [generate(task, n_fresh; rng=Xoshiro(s + EVAL_OFFSET)) for s in batch]
        Ys = [reduce(hcat, ms[2]) for ms in mon_sets]
        baselines = [target_baseline(ms[2]) for ms in mon_sets]

        trackers = [ConvergenceTracker(0.0; every=log_every, plateau_evals, min_improve)
                    for _ in batch]
        cbs = [tracker_callback(trackers[m],
            () -> normalized_eval(models[m], mon_sets[m][1], Ys[m], baselines[m];
                solver, reltol, abstol, dt);
            metric=maximum, lg, label="$(name)_s$(batch[m])") for m in eachindex(batch)]
        track = (E, epoch, errors) -> (foreach(cb -> cb(E, epoch, errors), cbs); nothing)
        on_epoch = combine_callbacks(track, ctx.save_cb)

        train_batched(models, gens; solver, n_epochs, num_samples,
            learning_rate=ctx.resume_lr, patience, lr_decay, min_lr, data_to_device, reltol, abstol, dt,
            eval_every=0, sync_every=log_every, verbose=0,
            on_loss=ctx.record_lr, on_epoch,
            backend, nsteps)

        if CUDA.functional()
            CUDA.synchronize()
            GC.gc()
            GC.gc()
            CUDA.reclaim()
        end

        eval_gens = [let inp = mon_sets[m][1], tgt = mon_sets[m][2]
                         () -> (inp, tgt)
                     end for m in eachindex(batch)]
        ek = batched_heldout_errors(models, eval_gens;
            backend, nsteps, solver, reltol, abstol, dt, data_to_device)

        for m in eachindex(batch)
            s = batch[m]
            tv = sum(baselines[m])
            in_d[s] = ek[m] / tv
            fresh_d[s] = ek[m] / tv
            conv_d[s] = converged(trackers[m])
            e2τ_d[s] = trackers[m].epoch_converged
            track_d[s] = track_matrix(trackers[m], length(baselines[m]))
            if rep[] === nothing
                decomp = (in_training=ek[m],
                    near_training=Pair{Float64,Float64}[], fresh=ek[m])
                rep[] = (decomp=decomp, baseline=baselines[m],
                    track_epochs=trackers[m].epochs,
                    track=track_matrix(trackers[m], length(baselines[m])),
                    errors=[maximum(v) for v in trackers[m].values],
                    n_params=n_params(models[m]))
            end
        end
        CUDA.functional() && (GC.gc(); CUDA.reclaim())
    end

    t0 = time()
    run_seed_sweep(seeds; experiment, tag=name, seed_batch, per_seed_schedule=false,
        base_lr=learning_rate, checkpoint_every=CHECKPOINT_EVERY, lg, label=name,
        init_model=(s -> init_model(task, hidden_dim, ncp; sigma, RepType, init_scale=0.3, T,
            rng=Xoshiro(s))),
        process,
        init_models=(warm === nothing ? nothing : warm.weights),
        init_lr=(warm === nothing ? nothing : warm.lr))
    wall = time() - t0

    in_s = [in_d[s] for s in seeds]
    fresh_s = [fresh_d[s] for s in seeds]
    conv_s = [conv_d[s] for s in seeds]
    e2τ_s = Union{Int,Nothing}[e2τ_d[s] for s in seeds]
    track_s = [get(track_d, s, nothing) for s in seeds]
    repr = rep[]
    in_norm = mean(in_s)
    fresh_norm = mean(fresh_s)
    conv_frac = mean(conv_s)
    e2τ_hit = [e for e in e2τ_s if e !== nothing]
    logln(lg, @sprintf("Done %s: in=%.4f±%.4f fresh=%.4f±%.4f conv=%d/%d wall=%.0fs",
        name, in_norm, std(in_s; corrected=false), fresh_norm,
        std(fresh_s; corrected=false), count(conv_s), M, wall))

    config = (sigma=Symbol(sigma), rep=string(nameof(RepType)),
        solver=string(nameof(typeof(solver))), dt=dt, T=string(T),
        backend=string(backend), nsteps=nsteps,
        learning_rate=learning_rate, num_samples=num_samples,
        seeds=collect(seeds), seed_batch=seed_batch, suffix=suffix)
    result = (task=task_name(task), N=N, h=hidden_dim, ncp=ncp,
        n_params=repr.n_params, τ=0.0, n_epochs=n_epochs,
        converged=conv_frac >= 0.5,
        epochs_to_τ=isempty(e2τ_hit) ? nothing : round(Int, median(e2τ_hit)),
        in_training_norm=in_norm, fresh_norm=fresh_norm,
        n_seeds=M, seeds=collect(seeds), converged_frac=conv_frac,
        in_training_std=std(in_s; corrected=false), fresh_std=std(fresh_s; corrected=false),
        per_seed=(in_training_norm=in_s, fresh_norm=fresh_s, converged=conv_s, epochs_to_τ=e2τ_s, track=track_s),
        decomposition=repr.decomp, baseline=repr.baseline,
        track_epochs=repr.track_epochs, track=repr.track,
        wall=wall, errors=repr.errors, config=config)
    save_run(path; result=result)
    return result
end

function phase_diagram(; tasks=TASKS, Ns=NS, hs=HS, ncps=NCPS,
    experiment::String=EXPERIMENT, lg=nothing, kwargs...)

    cells = [(task, N, h, ncp) for task in tasks for N in Ns for h in hs for ncp in ncps]
    logln(lg, "Phase Diagram: $(length(tasks)) tasks × " *
              "$(length(Ns))·$(length(hs))·$(length(ncps)) grid = $(length(cells)) cells")
    thunks = [() -> phase_cell(task; N, hidden_dim=h, ncp, experiment, lg, kwargs...)
              for (task, N, h, ncp) in cells]
    results = run_parallel(thunks; max_concurrent=1)

    logln(lg, "Phase Summary")
    for task in tasks
        tn = task_name(task)
        rs = [r for r in results if r.task == tn]
        med = isempty(rs) ? NaN : sort([r.fresh_norm for r in rs])[cld(length(rs), 2)]
        logln(lg, @sprintf("  %-28s %d cells, median fresh E/Var=%.3f", tn, length(rs), med))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    lg = TeeLog(EXPERIMENT, "sweep")
    phase_diagram(lg=lg; data_to_device=gpu_device(USE_GPU))
    close(lg)
end
