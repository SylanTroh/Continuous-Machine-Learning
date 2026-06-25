#!/usr/bin/env julia
# Usage:
#   OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/det_plateau.jl [n ...]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "setup", "gpu_env.jl"))
using NeuralFlow
using Printf
using Random: Xoshiro
using Statistics: mean, std
using LinearAlgebra: BLAS
using DifferentialEquations: Tsit5
using CUDA, Adapt
include(joinpath(@__DIR__, "cuda_utils.jl"))
include(joinpath(@__DIR__, "batched_sweep.jl"))

BLAS.set_num_threads(1)

const DET_SIZES = [2, 3, 4, 5]
const HIDDEN = 64
const NCP = 8
const NSTEPS = 8
const BACKEND = :fastrk4
const SIGMA = tanh
const REP = ChebPoly
const SOLVER = Tsit5()

const EXPERIMENT = get(ENV, "DP_EXP", "det_plateau")
const USE_GPU = get(ENV, "DP_GPU", "auto")

const NSEEDS = 100
const SEEDS = collect(1:NSEEDS)
const SEED_BATCH = 100
const EVAL_N = 1024

const NSAMP = 256
const NEPOCHS = 40000
const LOG_EVERY = 250
const CHECKPOINT_EVERY = 4 * LOG_EVERY
const RELTOL = 1e-3
const ABSTOL = 1e-6
const LR = 3e-3
const LR_DECAY = 0.75
const PATIENCE = 500
const MIN_LR = 1e-6
const PRECISION = Float32

function task_curves(task, seeds; data_to_device, lg=nothing, label::AbstractString="",
    experiment::AbstractString, tag::AbstractString)
    errs = Dict{Int,Float64}()
    vars = Dict{Int,Float64}()

    warm = load_models(experiment, tag)
    warm !== nothing &&
        logln(lg, "  [$label] warm-starting $(count(!isnothing, warm.weights))/$(length(seeds)) seeds")

    process = ctx -> begin
        batch = ctx.batch
        models = ctx.models
        gens = [
            let r = Xoshiro(s + DATA_OFFSET)
                () -> generate(task, NSAMP; rng=r)
            end for s in batch
        ]

        train_batched(models, gens; solver=SOLVER, n_epochs=NEPOCHS, num_samples=NSAMP,
            learning_rate=ctx.resume_lr, patience=PATIENCE, lr_decay=LR_DECAY, min_lr=MIN_LR,
            data_to_device, reltol=RELTOL, abstol=ABSTOL, dt=nothing,
            eval_every=0, sync_every=LOG_EVERY, verbose=0, backend=BACKEND, nsteps=NSTEPS,
            on_loss=ctx.record_lr, on_epoch=ctx.save_cb)

        if CUDA.functional()
            CUDA.synchronize()
            GC.gc()
            GC.gc()
            CUDA.reclaim()
        end

        eval_sets = [generate(task, EVAL_N; rng=Xoshiro(s + EVAL_OFFSET)) for s in batch]
        eval_gens = [let d = eval_sets[m]
                         () -> d
                     end for m in eachindex(batch)]
        ek = batched_heldout_errors(models, eval_gens; backend=:sciml, solver=Tsit5(),
            reltol=RELTOL, abstol=ABSTOL, data_to_device)
        for (m, s) in enumerate(batch)
            Y = reduce(hcat, eval_sets[m][2])
            v = only(target_baseline(Y))
            vars[s] = v
            errs[s] = ek[m] / v
        end
        CUDA.functional() && (GC.gc(); CUDA.reclaim())
    end

    res = run_seed_sweep(seeds; experiment, tag, seed_batch=SEED_BATCH, per_seed_schedule=false,
        base_lr=LR, checkpoint_every=CHECKPOINT_EVERY, lg, label,
        init_model=(s -> init_model(task, HIDDEN, NCP; sigma=SIGMA, RepType=REP,
            init_scale=0.3, T=PRECISION, rng=Xoshiro(s))),
        process,
        init_models=(warm === nothing ? nothing : warm.weights),
        init_lr=(warm === nothing ? nothing : warm.lr))
    samples = [c ./ NSAMP for c in res.per_seed_curves]
    return [errs[s] for s in seeds], [vars[s] for s in seeds], samples
end

function run_det_cell(n; data_to_device, lg=nothing)
    task = DeterminantTask(n)
    tag = "det$(n)"
    path = cell_path(EXPERIMENT, tag)
    if isfile(path) && !resume_enabled()
        logln(lg, "$tag cached, skipping")
        return load_run(path).summary
    end
    label = task_name(task)
    logln(lg, "$label  H$(HIDDEN) ncp$(NCP) nsteps$(NSTEPS) backend=$(BACKEND), $NSEEDS seeds, $NEPOCHS epochs")
    t0 = time()
    evars, vars, samples = task_curves(task, SEEDS; data_to_device, lg, label, experiment=EXPERIMENT, tag)
    nck = maximum(length, samples; init=0)
    sample_epochs = collect(LOG_EVERY:LOG_EVERY:(LOG_EVERY * nck))
    summary = (experiment=EXPERIMENT, task=label, n=n, hidden=HIDDEN, ncp=NCP, nsteps=NSTEPS,
        backend=string(BACKEND), T=string(PRECISION), nseeds=length(evars),
        mean=mean(evars), std=std(evars), evars=evars, vars=vars,
        samples=samples, sample_epochs=sample_epochs, num_samples=NSAMP)
    logln(lg, @sprintf("  %s  final E/Var mean=%.4f std=%.4f  (%.0fs)",
        label, summary.mean, summary.std, time() - t0))
    save_run(path; summary)
    return summary
end

function run_sweep(args)
    sizes = isempty(args) ? DET_SIZES : [parse(Int, a) for a in args]
    data_to_device = gpu_device(USE_GPU)
    lg = TeeLog(EXPERIMENT, "run")
    logln(lg, "det_plateau H$(HIDDEN) ncp$(NCP) nsteps$(NSTEPS), sizes=$(sizes), $(NSEEDS) seeds, $(NEPOCHS) epochs, T=$PRECISION, GPU=$(CUDA.functional())")
    logln(lg, "threads = $(Threads.nthreads())")
    t0 = time()
    for n in sizes
        run_det_cell(n; data_to_device, lg)
    end
    logln(lg, @sprintf("finished in %.0fs", time() - t0))
    close(lg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep(ARGS)
end
