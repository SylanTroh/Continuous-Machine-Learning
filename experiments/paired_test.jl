#!/usr/bin/env julia
# Usage: OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/paired_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "setup", "gpu_env.jl"))
using NeuralFlow
using Random
using Printf
using LinearAlgebra: BLAS
using CUDA, Adapt
include(joinpath(@__DIR__, "cuda_utils.jl"))
include(joinpath(@__DIR__, "batched_sweep.jl"))

BLAS.set_num_threads(1)

const EVAL_N = 1024
const ALPHA = 1e-3
const NSEEDS = 100
const SEEDS = collect(1:NSEEDS)

const EXPERIMENT = get(ENV, "PAIRED_EXP", "paired_test")
const USE_GPU = get(ENV, "PAIRED_GPU", "auto")
const BACKEND = :fastrk4

const RELTOL = 1e-3
const ABSTOL = 1e-6
const PATIENCE = 500
const LR_DECAY = 0.75
const MIN_LR = 1e-6
const LR = 3e-3
const NEPOCHS = 20000

const PRECISION = Float32
const HIDDEN = 64
const NCP = [4, 8, 16, 32]
const SEED_BATCH = 100
const NSAMP = 256

const LOG_EVERY = 250
const SAMPLE_EVERY = LOG_EVERY
const CHECKPOINT_EVERY = 4 * SAMPLE_EVERY
const PLATEAU_EVALS = 6
const MIN_IMPROVE = 0.01

const SPLITS = [
    (name="Spline", rep=Spline, backend=BACKEND),
    (name="ChebPoly", rep=ChebPoly, backend=BACKEND),
]

const PAIRED_CFG = (; nepochs=NEPOCHS, nsamp=NSAMP, lr=LR, patience=PATIENCE,
    lr_decay=LR_DECAY, min_lr=MIN_LR, reltol=RELTOL, abstol=ABSTOL, eval_n=EVAL_N,
    seed_batch=SEED_BATCH, T=PRECISION, sample_every=SAMPLE_EVERY,
    checkpoint_every=CHECKPOINT_EVERY, log_every=LOG_EVERY,
    plateau_evals=PLATEAU_EVALS, min_improve=MIN_IMPROVE)

function run_config(cfg, tag; splits=SPLITS, data_to_device, lg)
    path = cell_path(EXPERIMENT, tag)
    if isfile(path) && !resume_enabled()
        logln(lg, "$tag cached, skipping")
        return load_run(path).summary
    end
    task = cfg.make_task()
    names = [sp.name for sp in splits]
    logln(lg, "$(cfg.label): $NSEEDS seeds, reps = $(join(names, ", ")), T=$PRECISION backend=$BACKEND")
    logln(lg, "  threads = $(Threads.nthreads()), GPU = $(gpu_enabled(USE_GPU))")
    t0 = time()
    r = run_split_sweep(task, splits; seeds=SEEDS, hidden=cfg.hidden, ncp=cfg.ncp,
        experiment=EXPERIMENT, tag, data_to_device, lg, per_seed_schedule=false, cfg=PAIRED_CFG,
        cleanup=() -> (gpu_enabled(USE_GPU) && (GC.gc(); CUDA.reclaim())))
    logln(lg, @sprintf("  finished in %.0fs", time() - t0))
    means = [sum(e) / length(e) for e in r.errors]
    for (nm, m) in zip(r.names, means)
        logln(lg, @sprintf("  %-12s mean E = %.6e", nm, m))
    end
    summary = (name=cfg.label, reps=r.names, means=means,
        anova=(NSEEDS >= 2 ? anova_rm(r.errors) : nothing))
    sample_epochs = collect(SAMPLE_EVERY:SAMPLE_EVERY:NEPOCHS)
    meta = (label=cfg.label, hidden=cfg.hidden, ncp=cfg.ncp,
        reps=r.names, nsteps=euler_steps(cfg.ncp), backend=string(BACKEND),
        T=string(PRECISION), nepochs=NEPOCHS, nsamp=NSAMP, eval_n=EVAL_N,
        lr=LR, reltol=RELTOL, abstol=ABSTOL,
        plateau_evals=PLATEAU_EVALS, min_improve=MIN_IMPROVE,
        sample_every=SAMPLE_EVERY, sample_epochs=sample_epochs)
    save_run(path; summary, errors=r.errors, samples=r.samples, epochs=r.epochs,
        reps=r.names, config=meta)
    return summary
end

function make_configs()
    tasks = [("det(3)", () -> DeterminantTask(3)), ("MNIST", () -> load_mnist())]
    cfgs = [(label="$(tname) H$(HIDDEN) N$(N)", make_task=mk, ncp=N, hidden=HIDDEN)
            for (tname, mk) in tasks for N in NCP]
    return [(idx=i, c...) for (i, c) in enumerate(cfgs)]
end

function run_sweep(args)
    configs = make_configs()
    idxs = isempty(args) ? collect(eachindex(configs)) : parse.(Int, args)
    data_to_device = gpu_device(USE_GPU)
    lg = TeeLog(EXPERIMENT, "config_" * join(idxs, "_"))
    for i in idxs
        run_config(configs[i], "config_$(configs[i].idx)"; data_to_device, lg)
    end
    close(lg)
end

function paired_report(load, lg)
    pct = @sprintf("%g", 100 * (1 - ALPHA))
    for cfg in make_configs()
        nt = load("config_$(cfg.idx)")
        if nt === nothing
            logln(lg, @sprintf("  config_%d  (missing)", cfg.idx))
            continue
        end
        errors = [Float64.(e) for e in nt.errors]
        logln(lg, "$(nt.summary.name):")
        for (nm, e) in zip(nt.reps, errors)
            ci = mean_ci(e; level=1 - ALPHA)
            logln(lg, @sprintf("    %-12s mean E = %.4e   [%.4e, %.4e]", nm, ci.mean, ci.lo, ci.hi))
        end
        if length(errors[1]) >= 2
            a = anova_rm(errors)
            logln(lg, @sprintf("    F(%d,%d) = %.2f, p = %.2g, partial η² = %.3f",
                a.df_rep, a.df_resid, a.F, a.p, a.partial_eta2))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "report"
        report(paired_report, EXPERIMENT; save=("save" in ARGS))
    else
        run_sweep(ARGS)
    end
end
