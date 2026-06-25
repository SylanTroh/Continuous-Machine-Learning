#!/usr/bin/env julia
# Usage:
#   OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/cell_depth.jl [arm-label ...]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "setup", "gpu_env.jl"))
using NeuralFlow
using Printf
using Random: Xoshiro
using Statistics: mean, std
using LinearAlgebra: BLAS
using CUDA, Adapt
include(joinpath(@__DIR__, "cuda_utils.jl"))

BLAS.set_num_threads(1)

const EXPERIMENT = get(ENV, "CD_EXP", "cell_depth")
const USE_GPU = get(ENV, "CD_GPU", "auto")

const DET_N = 3
const DET_H = 16
const MNIST_H = 64
const BACKEND = :fastrk4
const SIGMA = tanh
const REP = Spline

const ARMS = [
    ("L1",         Rational{Int}[1//1],             16),
    ("L2",         Rational{Int}[1//1, 1//1],       16),
    ("bottleneck", Rational{Int}[1//4, 1//4, 1//1], 16),
    ("L2",         Rational{Int}[1//1, 1//1],        8),
]
const NCPS = [1, 2, 4, 8]

const NSEEDS = 100
const SEEDS = collect(1:NSEEDS)
const NSAMP = 256
const NEPOCHS = 10000
const EVAL_N = 1024
const LR = 2e-3
const LR_DECAY = 0.75
const PATIENCE = 500
const MIN_LR = 1e-6
const PRECISION = Float32
const CHECKPOINT_EVERY = 1000

_stack3(sets, sel) = cat([PRECISION.(reduce(hcat, [vec(x) for x in s[sel]])) for s in sets]...; dims=3)

function build_tasks(which)
    out = NamedTuple[]
    if "det3" in which
        t = DeterminantTask(DET_N)
        push!(out, (name="det$(DET_N)", hidden=DET_H, train=t, eval=t))
    end
    if "mnist" in which
        push!(out, (name="mnist", hidden=MNIST_H,
            train=load_mnist(split=:train), eval=load_mnist(split=:test)))
    end
    return out
end

function heldout_metrics(eval_task, models, nsteps; data_to_device)
    eval_sets = [generate(eval_task, EVAL_N; rng=Xoshiro(s + EVAL_OFFSET)) for s in SEEDS]
    P = data_to_device(block_stack_group(models))
    Ains = data_to_device(_stack3(eval_sets, 1))
    Z = block_fast_group_predict(P, Ains, models[1]; nsteps=nsteps, method=:rk4)
    evars = Float64[]
    accs = Float64[]
    for (k, _) in enumerate(SEEDS)
        Y = PRECISION.(reduce(hcat, [vec(t) for t in eval_sets[k][2]]))
        Zk = Z[:, :, k]
        sse = sum(abs2, Zk .- Y)
        v = sum(target_baseline(Y))
        push!(evars, (sse / EVAL_N) / v)
        push!(accs, mean(argmax(view(Zk, :, n)) == argmax(view(Y, :, n)) for n in axes(Y, 2)))
    end
    return evars, accs
end

function run_cell(tk, name, mults, ncp, nsteps; data_to_device, lg)
    tag = "$(tk.name)_$(name)_H$(tk.hidden)_ncp$(ncp)_nsteps$(nsteps)"
    path = cell_path(EXPERIMENT, tag)
    if isfile(path) && !resume_enabled()
        logln(lg, "$tag cached, skipping")
        return
    end
    warm = load_models(EXPERIMENT, tag)
    models = warm !== nothing ? warm :
             [init_resblock(tk.train, tk.hidden, ncp, mults; sigma=SIGMA, RepType=REP,
                  init_scale=0.3, T=PRECISION, rng=Xoshiro(s)) for s in SEEDS]
    warm !== nothing && logln(lg, "  [$tag] warm-starting $(length(models)) seeds")
    gens = [let r = Xoshiro(s + DATA_OFFSET)
                () -> generate(tk.train, NSAMP; rng=r)
            end for s in SEEDS]
    save_cb = ms -> save_models(EXPERIMENT, tag, [copy_model(m) for m in ms])

    logln(lg, "$tag spec=$(mults), $NSEEDS seeds, $NEPOCHS epochs")
    t0 = time()
    models, errs = train_block_batched(models, gens; backend=BACKEND, nsteps=nsteps,
        n_epochs=NEPOCHS, num_samples=NSAMP, learning_rate=LR, patience=PATIENCE,
        lr_decay=LR_DECAY, min_lr=MIN_LR, verbose=0, data_to_device,
        loss=squared_error_loss, checkpoint_every=CHECKPOINT_EVERY, checkpoint_fn=save_cb)

    if CUDA.functional()
        CUDA.synchronize(); GC.gc(); GC.gc(); CUDA.reclaim()
    end
    save_cb(models)

    evars, accs = heldout_metrics(tk.eval, models, nsteps; data_to_device)
    CUDA.functional() && (GC.gc(); CUDA.reclaim())

    nparam = n_params(models[1])
    summary = (experiment=EXPERIMENT, task=tk.name, hidden=tk.hidden, spec=name,
        mults=string(mults), ncp=ncp, nsteps=nsteps, nlayers=length(mults), nparams=nparam,
        backend=string(BACKEND), T=string(PRECISION), nseeds=length(evars),
        evar_mean=mean(evars), evar_std=std(evars), acc_mean=mean(accs), acc_std=std(accs),
        evars=evars, accs=accs, train_curve=errs, num_samples=NSAMP)
    logln(lg, @sprintf("  %s  nparams=%d  E/Var=%.4f±%.4f  acc=%.4f±%.4f  (%.0fs)",
        tag, nparam, summary.evar_mean, summary.evar_std, summary.acc_mean, summary.acc_std, time() - t0))
    save_run(path; summary)
    return
end

function run_sweep(args)
    tasks = build_tasks(["det3", "mnist"])
    isempty(tasks) && error("no tasks selected")
    chosen = isempty(args) ? ARMS : [a for a in ARMS if a[1] in args]
    isempty(chosen) && error("no matching arms in $(args); known: $(unique(first.(ARMS)))")
    data_to_device = gpu_device(USE_GPU)
    lg = TeeLog(EXPERIMENT, "run")
    logln(lg, "cell_depth tasks=$([t.name for t in tasks]) backend=$(BACKEND), " *
              "arms=$([(a[1], a[3]) for a in chosen]) × ncp=$(NCPS), $(NSEEDS) seeds, $(NEPOCHS) epochs, T=$PRECISION, GPU=$(CUDA.functional())")
    logln(lg, "threads = $(Threads.nthreads())")
    t0 = time()
    for tk in tasks, (name, mults, nsteps) in chosen, ncp in NCPS
        run_cell(tk, name, mults, ncp, nsteps; data_to_device, lg)
    end
    logln(lg, @sprintf("finished in %.0fs", time() - t0))
    close(lg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep(ARGS)
end
