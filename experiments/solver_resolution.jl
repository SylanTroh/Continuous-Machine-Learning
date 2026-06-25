#!/usr/bin/env julia
# Usage:
#   OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/solver_resolution.jl [split ...]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "setup", "gpu_env.jl"))
using NeuralFlow
using Printf
using Random: Xoshiro
using Statistics: mean, std
using LinearAlgebra: BLAS
using DifferentialEquations: Tsit5
using OrdinaryDiffEqLowOrderRK: Euler
using CUDA, Adapt
include(joinpath(@__DIR__, "cuda_utils.jl"))
include(joinpath(@__DIR__, "batched_sweep.jl"))

BLAS.set_num_threads(1)

const TASK = DeterminantTask(3)
const REP = ChebPoly
const SIGMA = tanh
const HIDDEN = 64
const K_GRID = [4, 8, 16, 32, 64]
const FIXED_NCP = 8

const EXPERIMENT = get(ENV, "SR_EXP", "solver_resolution")
const USE_GPU = get(ENV, "SR_GPU", "auto")

const EVAL_N = 1024
const NSEEDS = 100
const SEEDS = collect(1:NSEEDS)
const SEED_BATCH = 100

const RELTOL = 1e-3
const ABSTOL = 1e-6
const PATIENCE = 500
const LR_DECAY = 0.75
const MIN_LR = 1e-6
const LR = 3e-3
const NEPOCHS = 20000
const NSAMP = 256
const LOG_EVERY = 250
const CHECKPOINT_EVERY = 4 * LOG_EVERY
const PRECISION = Float32

const SPLITS = [
    (name=:euler_ncp, label="Euler ncp=nsteps"),
    (name=:euler_steps, label="Euler nsteps (ncp=$(FIXED_NCP))"),
    (name=:rk_ncp, label="RK4 ncp=nsteps"),
    (name=:rk_steps, label="RK4 nsteps (ncp=$(FIXED_NCP))"),
]

function split_kwargs(split::Symbol, k::Int)
    euler = (backend=:fasteuler, solver=Euler())
    rk = (backend=:fastrk4, solver=Tsit5(), dt=nothing)
    if split === :euler_ncp
        (ncp=k, nsteps=k, dt=1.0 / k, euler...)
    elseif split === :euler_steps
        (ncp=FIXED_NCP, nsteps=k, dt=1.0 / k, euler...)
    elseif split === :rk_ncp
        (ncp=k, nsteps=k, rk...)
    elseif split === :rk_steps
        (ncp=FIXED_NCP, nsteps=k, rk...)
    else
        error("unknown split $split")
    end
end

function cell_evars(kw, seeds; data_to_device, lg=nothing, label::AbstractString="",
    experiment::AbstractString, tag::AbstractString)
    errs = Dict{Int,Float64}()

    warm = load_models(experiment, tag)
    warm !== nothing &&
        logln(lg, "  [$label] warm-starting $(count(!isnothing, warm.weights))/$(length(seeds)) seeds")

    process = ctx -> begin
        batch = ctx.batch
        models = ctx.models
        gens = [
            let r = Xoshiro(s + DATA_OFFSET)
                () -> generate(TASK, NSAMP; rng=r)
            end for s in batch
        ]

        train_batched(models, gens; solver=kw.solver, n_epochs=NEPOCHS, num_samples=NSAMP,
            learning_rate=ctx.resume_lr, patience=PATIENCE, lr_decay=LR_DECAY, min_lr=MIN_LR,
            data_to_device, reltol=RELTOL, abstol=ABSTOL, dt=kw.dt,
            eval_every=0, sync_every=LOG_EVERY, verbose=0, backend=kw.backend, nsteps=kw.nsteps,
            on_loss=ctx.record_lr, on_epoch=ctx.save_cb)

        if CUDA.functional()
            CUDA.synchronize()
            GC.gc()
            GC.gc()
            CUDA.reclaim()
        end

        eval_sets = [generate(TASK, EVAL_N; rng=Xoshiro(s + EVAL_OFFSET)) for s in batch]
        eval_gens = [let d = eval_sets[m]
                         () -> d
                     end for m in eachindex(batch)]
        ek = batched_heldout_errors(models, eval_gens; backend=:sciml, solver=Tsit5(),
            reltol=RELTOL, abstol=ABSTOL, data_to_device)
        for (m, s) in enumerate(batch)
            Y = reduce(hcat, eval_sets[m][2])
            errs[s] = ek[m] / only(target_baseline(Y))
        end
        CUDA.functional() && (GC.gc(); CUDA.reclaim())
    end

    res = run_seed_sweep(seeds; experiment, tag, seed_batch=SEED_BATCH, per_seed_schedule=false,
        base_lr=LR, checkpoint_every=CHECKPOINT_EVERY, lg, label,
        init_model=(s -> init_model(TASK, HIDDEN, kw.ncp; sigma=SIGMA, RepType=REP,
            init_scale=0.3, T=PRECISION, rng=Xoshiro(s))),
        process,
        init_models=(warm === nothing ? nothing : warm.weights),
        init_lr=(warm === nothing ? nothing : warm.lr))
    samples = [c ./ NSAMP for c in res.per_seed_curves]
    return [errs[s] for s in seeds], samples
end

function run_split_cell(split, k; data_to_device, lg=nothing)
    tag = "$(split.name)_k$(k)"
    path = cell_path(EXPERIMENT, tag)
    if isfile(path) && !resume_enabled()
        logln(lg, "$tag cached, skipping")
        return load_run(path).summary
    end
    kw = split_kwargs(split.name, k)
    label = "$(split.label) k$(k)"
    logln(lg, "$label  (ncp=$(kw.ncp), nsteps=$(kw.nsteps), backend=$(kw.backend)), $NSEEDS seeds")
    t0 = time()
    evars, samples = cell_evars(kw, SEEDS; data_to_device, lg, label, experiment=EXPERIMENT, tag)
    nck = maximum(length, samples; init=0)
    sample_epochs = collect(LOG_EVERY:LOG_EVERY:(LOG_EVERY * nck))
    summary = (experiment=EXPERIMENT, split=split.name, label=split.label, k=k,
        ncp=kw.ncp, nsteps=kw.nsteps, backend=string(kw.backend), T=string(PRECISION),
        n=length(evars), mean=mean(evars), std=std(evars), evars=evars,
        samples=samples, sample_epochs=sample_epochs, num_samples=NSAMP)
    logln(lg, @sprintf("  %s  E/Var mean=%.4f std=%.4f  (%.0fs)",
        label, summary.mean, summary.std, time() - t0))
    save_run(path; summary)
    return summary
end

const REPORT_LEVEL = 0.999
const BASELINE_K = FIXED_NCP
const SPLIT_NAMES = [:euler_ncp, :euler_steps, :rk_ncp, :rk_steps]

function solver_report(load, lg)
    pct = @sprintf("%g", 100 * REPORT_LEVEL)
    evars(split, k) = (nt = load("$(split)_k$(k)"); nt === nothing ? nothing : nt.summary.evars)
    splabel(split) = (nt = load("$(split)_k$(BASELINE_K)"); nt === nothing ? string(split) : nt.summary.label)

    logln(lg, "solver_resolution, $(task_name(TASK)) H$(HIDDEN): held-out Ê = E/Var, paired difference vs the k=$(BASELINE_K) baseline, $(pct)% CI (n=$(NSEEDS))")
    for split in SPLIT_NAMES
        logln(lg, "")
        logln(lg, "  $(splabel(split))")
        logln(lg, @sprintf("    %-4s %10s %16s %26s", "k", "Ê", "Ê−Ê(k=$BASELINE_K)", "$(pct)% CI"))
        base = evars(split, BASELINE_K)
        for k in K_GRID
            cells = evars(split, k)
            if cells === nothing || base === nothing
                logln(lg, @sprintf("    %-4d  (incomplete)", k))
                continue
            end
            if k == BASELINE_K
                logln(lg, @sprintf("    %-4d %10.4f %16s %26s", k, mean(cells), "-", "- (baseline)"))
            else
                d = mean_ci(cells .- base; level=REPORT_LEVEL)
                logln(lg, @sprintf("    %-4d %10.4f %+16.4f   [%+.4f, %+.4f]",
                    k, mean(cells), d.mean, d.lo, d.hi))
            end
        end
    end
end

function run_sweep(args)
    splits = isempty(args) ? SPLITS : filter(sp -> string(sp.name) in args, SPLITS)
    isempty(splits) && error("no splits match $args; known: $(join(string.(sp.name for sp in SPLITS), ", "))")
    data_to_device = gpu_device(USE_GPU)
    lg = TeeLog(EXPERIMENT, "run")
    logln(lg, "$(task_name(TASK)) H$(HIDDEN), $(NSEEDS) seeds, k = $(K_GRID), T=$PRECISION, GPU=$(CUDA.functional())")
    logln(lg, "threads = $(Threads.nthreads())")
    t0 = time()
    for split in splits, k in K_GRID
        run_split_cell(split, k; data_to_device, lg)
    end
    logln(lg, @sprintf("finished in %.0fs", time() - t0))
    close(lg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ARGS, 1, "") == "report"
        report(solver_report, EXPERIMENT; save=("save" in ARGS))
    else
        run_sweep(ARGS)
    end
end
