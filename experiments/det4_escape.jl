#!/usr/bin/env julia
# Usage:  OMP_NUM_THREADS=1 julia --project=. --threads=1 experiments/det4_escape.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "setup", "gpu_env.jl"))

using NeuralFlow
using Random
using Printf
using Statistics: mean
using LinearAlgebra: BLAS
using CUDA, Adapt
include(joinpath(@__DIR__, "cuda_utils.jl"))
include(joinpath(@__DIR__, "batched_sweep.jl"))

BLAS.set_num_threads(1)

const DET_N = 4
const EXPERIMENT = get(ENV, "D4_EXP", "det$(DET_N)_escape")
const USE_GPU = get(ENV, "D4_GPU", "auto")

const TASK = DeterminantTask(DET_N)
const HIDDEN = 64
const NCP = 8
const NUM_SAMPLES = 128
const NSEEDS = 1
const SEEDS = collect(1:NSEEDS)
const SEED_BATCH = 1

const REP = ChebPoly
const SIGMA = relu
const PRECISION = Float32
const BACKEND = :fastrk4
const NSTEPS = NCP

const NEPOCHS = 5_000
const LR = 1e-3
const PATIENCE = 500
const LR_DECAY = 0.75
const MIN_LR = 1e-6
const ESCAPE_GATE = 0.9
const RELTOL = 1e-3
const ABSTOL = 1e-6

const EVAL_EVERY = 250
const LOG_EVERY = EVAL_EVERY
const CHECKPOINT_EVERY = 5000
const ESCAPE_FRAC = 0.5
const EXTRA_EPOCHS = 20_000

function det4_escape(; lg=nothing, data_to_device=identity)
    name = "$(task_name(TASK))_N$(NUM_SAMPLES)_h$(HIDDEN)_ncp$(NCP)"

    init_models = [init_model(TASK, HIDDEN, NCP; sigma=SIGMA, RepType=REP,
        init_scale=0.3, T=PRECISION, rng=Xoshiro(s)) for s in SEEDS]
    init_path = model_path(EXPERIMENT, name * "_init")
    if !isfile(init_path)
        save_run(init_path; models=(; weights=init_models, seeds=SEEDS))
        logln(lg, "saved initial weights ($(NSEEDS) seeds) → $init_path")
    end

    warm = load_models(EXPERIMENT, name)
    warm !== nothing &&
        logln(lg, "warm-starting $(count(!isnothing, warm.weights))/$(NSEEDS) seeds from checkpoint")

    escape_log = Dict{Int,Int}()

    process = ctx -> begin
        batch = ctx.batch
        models = ctx.models
        K = length(batch)
        gens = [
            let r = Xoshiro(s + DATA_OFFSET)
                () -> generate(TASK, NUM_SAMPLES; rng=r)
            end for s in batch
        ]

        e_first = fill(NaN, K)
        e_best = fill(Inf, K)
        escaped = falses(K)
        escape_epoch = Ref{Union{Nothing,Int}}(nothing)

        on_eval = (it, perseed) -> begin
            best_ratio = Inf
            best_k = 1
            for k in 1:K
                ps = perseed[k] / NUM_SAMPLES
                isnan(e_first[k]) && (e_first[k] = ps)
                ps < e_best[k] && (e_best[k] = ps)
                r = e_best[k] / e_first[k]
                r < best_ratio && (best_ratio = r; best_k = k)
                if !escaped[k] && r < ESCAPE_FRAC
                    escaped[k] = true
                    escape_log[batch[k]] = it
                    escape_epoch[] === nothing && (escape_epoch[] = it)
                    logln(lg, @sprintf("  *** Escape seed %d @ epoch %d: per-sample E %.4e → %.4e (%.1f%% of first)",
                        batch[k], it, e_first[k], e_best[k], 100 * r))
                end
            end
            logln(lg, @sprintf("  epoch %7d  best seed %d ratio %.3f (E=%.4e)  escaped=%d/%d",
                it, batch[best_k], best_ratio, e_best[best_k], count(escaped), K))
            nothing
        end

        stop_cb = (E, epoch, errors) -> begin
            ee = escape_epoch[]
            (ee !== nothing && epoch >= ee + EXTRA_EPOCHS) || return nothing
            logln(lg, @sprintf("Stop @ epoch %d (%d epochs after first escape @ %d)",
                epoch, epoch - ee, ee))
            :stop
        end

        on_epoch = combine_callbacks(stop_cb, ctx.save_cb)

        train_batched(models, gens; n_epochs=NEPOCHS, num_samples=NUM_SAMPLES,
            learning_rate=ctx.resume_lr, patience=PATIENCE, lr_decay=LR_DECAY, min_lr=MIN_LR,
            data_to_device, reltol=RELTOL, abstol=ABSTOL, dt=nothing,
            eval_every=EVAL_EVERY, sync_every=LOG_EVERY, verbose=0,
            backend=BACKEND, nsteps=NSTEPS,
            on_loss=ctx.record_lr, on_epoch, on_eval,
            per_seed_schedule=true, plateau_evals=0, min_improve=0.0, escape_frac=ESCAPE_GATE,
            lr0=ctx.lr0, lr_out=ctx.lr_live)

        if CUDA.functional()
            CUDA.synchronize()
            GC.gc()
            GC.gc()
            CUDA.reclaim()
        end
    end

    t0 = time()
    run_seed_sweep(SEEDS; experiment=EXPERIMENT, tag=name, seed_batch=SEED_BATCH,
        per_seed_schedule=true, base_lr=LR, checkpoint_every=CHECKPOINT_EVERY, lg, label=name,
        init_model=(s -> init_model(TASK, HIDDEN, NCP; sigma=SIGMA, RepType=REP,
            init_scale=0.3, T=PRECISION, rng=Xoshiro(s))),
        process,
        init_models=(warm === nothing ? init_models : warm.weights),
        init_lr=(warm === nothing ? nothing : warm.lr))
    wall = time() - t0

    esc = sort(collect(escape_log); by=last)
    result = (task=task_name(TASK), N=NUM_SAMPLES, h=HIDDEN, ncp=NCP, n_seeds=NSEEDS,
        escape_frac=ESCAPE_FRAC, extra_epochs=EXTRA_EPOCHS, n_escaped=length(esc),
        escapes=esc, wall=wall, config=(rep=string(nameof(REP)), sigma=Symbol(SIGMA),
            backend=string(BACKEND), nsteps=NSTEPS, lr=LR, precision=string(PRECISION)))
    save_run(cell_path(EXPERIMENT, name); result=result)
    logln(lg, @sprintf("DONE %s: %d/%d seeds escaped, wall=%.0fs",
        name, length(esc), NSEEDS, wall))
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    lg = TeeLog(EXPERIMENT, "run")
    logln(lg, "$(EXPERIMENT): $(task_name(TASK)) h=$HIDDEN ncp=$NCP $(NUM_SAMPLES)samp $(REP) $(SIGMA) " *
              "$(NSEEDS)seeds backend=$BACKEND nsteps=$NSTEPS lr=$LR nepochs=$NEPOCHS " *
              "escape<$(ESCAPE_FRAC)×first +$(EXTRA_EPOCHS)")
    det4_escape(; lg, data_to_device=gpu_device(USE_GPU))
    close(lg)
end
