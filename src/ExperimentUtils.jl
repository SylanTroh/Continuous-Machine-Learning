"""
Shared scaffolding for the experiment scripts
"""
module ExperimentUtils

using Printf
using Random
using Statistics
using Dates: format, now
using HypothesisTests: OneSampleTTest, pvalue, confint
using GLM: lm, r2, ftest, deviance, @formula
using DifferentialEquations: Tsit5
using ..Tasks: AbstractTask, generate, task_name
using ..SplineRepresentation: Spline
using ..TrainingUtils: compute_error
using ..ModelInit: init_model
using ..Adjoint: train_adjoint, batched_predict, _default_sensealg
using ..FixedTrainingData: DataPool, generate_sampler
using ..Results: experiment_dir, save_run, load_run
using ..GradientUtils: CurveSink, with_curve_sink, take_curves

export run_parallel
export TeeLog, logln, loss_logger
export _envint, euler_steps
export cell_path, run_cell, seed_cell, report
export model_path, ckpt_path, save_models, load_models, resume_enabled
export combine_callbacks, periodic_save_models
export num_concurrent, paired_ttest, mean_ci, anova_rm
export target_baseline, normalized_eval, heldout_error
export ConvergenceTracker, tracker_callback, converged, plateaued, final_values, track_matrix
export train_run, tracked_train_run
export DATA_OFFSET, EVAL_OFFSET

const DATA_OFFSET = 100000
const EVAL_OFFSET = 200000

"""
    run_parallel(thunks; max_concurrent=2) → Vector

Run argument-less functions concurrently, at most `max_concurrent` at a time.
"""
function run_parallel(thunks; max_concurrent::Int=4)
    sem = Base.Semaphore(max_concurrent)
    tasks = map(enumerate(thunks)) do (i, thunk)
        Threads.@spawn begin
            Base.acquire(sem)
            try
                thunk()
            catch e
                println("run_parallel: task $i failed: ", sprint(showerror, e))
                flush(stdout)
                rethrow()
            finally
                Base.release(sem)
            end
        end
    end
    return map(fetch, tasks)
end

_ts() = format(now(), "HH:MM:SS")

"""
    TeeLog(experiment, run)
    TeeLog(path)

Log sink writing to both stdout and `results/<experiment>/<run>.log`
"""
struct TeeLog
    io::IOStream
end

function TeeLog(path::AbstractString)
    mkpath(dirname(path))
    return TeeLog(open(path, "a"))
end

TeeLog(experiment::AbstractString, run::AbstractString) =
    TeeLog(joinpath(experiment_dir(experiment), run * ".log"))

Base.close(lg::TeeLog) = close(lg.io)

"""
    logln(lg, msg)

Write a timestamped line to stdout, and to the log file.
"""
function logln(lg::Union{Nothing,TeeLog}, msg::AbstractString)
    line = "[$(_ts())] $msg"
    println(line)
    flush(stdout)
    if lg !== nothing
        println(lg.io, line)
        flush(lg.io)
    end
    return nothing
end

"""
    _envint(key, default) → Int

Parse integer experiment `key` from the environment.
"""
_envint(key::AbstractString, default::Integer) = parse(Int, get(ENV, key, string(default)))

"""
    euler_steps(ncp) → Int
"""
euler_steps(ncp::Integer) = max(ncp - 1, 1)

"""
    loss_logger(lg) → (channel, task)

Async logging for threaded training
"""
function loss_logger(lg)
    ch = Channel{String}(Inf)
    task = Threads.@spawn for msg in ch
        logln(lg, msg)
    end
    return ch, task
end

"""
    cell_path(experiment, name) → String
"""
cell_path(experiment::AbstractString, name::AbstractString) =
    joinpath(experiment_dir(experiment), name * ".jls")

"""
    model_path(experiment, name) → String

Checkpoint path for a cell's trained weights:
`<experiment>/<name>.model.jls`.
"""
model_path(experiment::AbstractString, name::AbstractString) =
    cell_path(experiment, name * ".model")

"""
    ckpt_path(experiment, name) → String

Path for a cell's mid-training `Checkpointer`: `<experiment>/<name>.ckpt.jls`.
"""
ckpt_path(experiment::AbstractString, name::AbstractString) =
    cell_path(experiment, name * ".ckpt")

"""
    save_models(experiment, name, models) → String

Persist a cell's trained weights so a rerun can restart from the checkpoint.
"""
function save_models(experiment::AbstractString, name::AbstractString, models)
    p = model_path(experiment, name)
    save_run(p; models)
    return p
end

"""
    load_models(experiment, name) → models | nothing

Load a cell's weight checkpoint written by `save_models`, or `nothing` if absent.
"""
load_models(experiment::AbstractString, name::AbstractString) =
    (p = model_path(experiment, name); isfile(p) ? load_run(p).models : nothing)

"""
    seed_cell(experiment, tag, s) → String

Result path for one seed of a paired config: `<experiment>/<tag>_seeds/seed_<s>.jls`.
"""
seed_cell(experiment::AbstractString, tag::AbstractString, s::Integer) =
    cell_path(experiment, "$(tag)_seeds/seed_$(s)")

"""
    num_concurrent(default=8) → Int

Number of seeds to train at once.
"""
num_concurrent(default::Int=8) = parse(Int, get(ENV, "NUM_CONCURRENT", string(default)))

"""
    resume_enabled() → Bool

"""
resume_enabled() = get(ENV, "RESUME", "") == "1"

"""
    run_cell(f, path; lg=nothing) → NamedTuple
"""
function run_cell(f, path::AbstractString; lg=nothing)
    if isfile(path) && !resume_enabled()
        logln(lg, "SKIP $(basename(path))")
        return load_run(path)
    end
    isfile(path) && logln(lg, "RESUME $(basename(path))")

    sink = CurveSink()
    nt = with_curve_sink(() -> f(), sink)
    curves = [c.curve for c in take_curves(sink)]
    nt = (isempty(curves) || haskey(nt, :loss_curves)) ? nt : merge(nt, (; loss_curves=curves))
    save_run(path; nt...)
    return nt
end

"""
    combine_callbacks(cbs...) → Function | nothing

Merge several `on_epoch(E, epoch, errors)` callbacks into one.
"""
function combine_callbacks(cbs...)
    live = [cb for cb in cbs if cb !== nothing]
    isempty(live) && return nothing
    return (E, epoch, errors) -> begin
        stop = false
        for cb in live
            cb(E, epoch, errors) === :stop && (stop = true)
        end
        stop ? :stop : nothing
    end
end

"""
    periodic_save_models(experiment, name, every; payload, started) → Function | nothing
"""
function periodic_save_models(experiment::AbstractString, name::AbstractString,
    every::Integer; payload::Function, started::Ref{Bool})
    every > 0 || return nothing
    return (E, epoch, errors) -> begin
        epoch % every == 0 || return nothing
        save_run(model_path(experiment, name); overwrite=started[], models=payload())
        started[] = true
        return nothing
    end
end

"""
    report(reporter, experiment; save=false, lg=nothing) → Nothing

Print an experiment's results from its saved `results/<experiment>/`.
"""
function report(reporter::Function, experiment::AbstractString;
    save::Bool=false, lg::Union{Nothing,TeeLog}=nothing)
    own = lg === nothing && save
    own && (lg = TeeLog(experiment, "report"))
    load(name::AbstractString) = begin
        p = cell_path(experiment, name)
        isfile(p) ? load_run(p) : nothing
    end
    try
        reporter(load, lg)
    finally
        own && close(lg)
    end
    return nothing
end

"""
    paired_ttest(name, diffs; diff_label="", alpha=1e-3, lg=nothing) → NamedTuple

One-sample t-test on the paired differences `diffs`.
"""
function paired_ttest(name::AbstractString, diffs::Vector{Float64};
    diff_label::AbstractString="", alpha::Real=1e-3, lg=nothing)
    n = length(diffs)
    md = mean(diffs)
    lbl = isempty(diff_label) ? "" : " ($diff_label)"
    if n < 2
        # A t-test needs ≥2 paired observations
        logln(lg, "$name (n=$n)")
        logln(lg, @sprintf("  mean diff%s = %.4e  (t-test skipped: needs n ≥ 2)", lbl, md))
        return (name=name, n=n, mean=md, sd=NaN, t=NaN, df=0, p_t=NaN,
            ci=(NaN, NaN), cohen_dz=NaN, diffs=diffs)
    end
    sd = std(diffs)
    tt = OneSampleTTest(diffs)
    p = pvalue(tt)
    lo, hi = confint(tt; level=1 - alpha)
    dz = sd == 0 ? Inf : md / sd
    logln(lg, "$name (n=$n)")
    logln(lg, @sprintf("  mean diff%s = %.4e", lbl, md))
    logln(lg, @sprintf("  %g%% CI = [%.4e, %.4e]", 100 * (1 - alpha), lo, hi))
    logln(lg, @sprintf("  t = %.3f, df = %d, p = %.3g", tt.t, tt.df, p))
    logln(lg, @sprintf("  Cohen's dz = %.3f", dz))
    return (name=name, n=n, mean=md, sd=sd, t=tt.t, df=tt.df, p_t=p,
        ci=(lo, hi), cohen_dz=dz, diffs=diffs)
end

"""
    mean_ci(values; level=0.95) → (; n, mean, lo, hi)

One-sample t confidence interval
"""
function mean_ci(values::AbstractVector{<:Real}; level::Real=0.95)
    v = Float64.(values)
    m = mean(v)
    std(v) == 0 && return (; n=length(v), mean=m, lo=m, hi=m)
    lo, hi = confint(OneSampleTTest(v); level)
    return (; n=length(v), mean=m, lo, hi)
end

"""
    anova_rm(arms) → NamedTuple

ANOVA via the GLM library.
"""
function anova_rm(arms::AbstractVector{<:AbstractVector{<:Real}})
    k = length(arms)
    k >= 2 || error("anova_rm needs ≥2 conditions, got $k")
    n = length(arms[1])
    all(length(a) == n for a in arms) || error("all arms must be the same length (paired by index)")

    err = Float64[]
    rep = String[]
    seed = String[]
    for (c, a) in enumerate(arms), s in 1:n
        push!(err, Float64(a[s]))
        push!(rep, "c$c")
        push!(seed, "s$s")
    end
    tbl = (; err, rep, seed)

    m_rep = lm(@formula(err ~ rep), tbl)
    m_seed = lm(@formula(err ~ seed), tbl)
    m_full = lm(@formula(err ~ seed + rep), tbl)

    eta2 = r2(m_rep)
    seed_eta2 = r2(m_seed)

    ft = ftest(m_seed.model, m_full.model)
    F = last(ft.fstat)
    p = last(ft.pval)

    SS_resid = deviance(m_full)
    SS_rep = deviance(m_seed) - SS_resid
    partial_eta2 = SS_rep / (SS_rep + SS_resid)

    cond_means = [mean(Float64.(a)) for a in arms]
    return (; k, n, cond_means,
        eta2, seed_eta2, partial_eta2,
        SS_rep, SS_resid, df_rep=k - 1, df_resid=(k - 1) * (n - 1),
        F, p)
end

"""
    target_baseline(targets) → Vector{Float64}

Per-component variance of the targets
"""
target_baseline(targets::AbstractVector) = target_baseline(reduce(hcat, targets))
target_baseline(Y::AbstractMatrix) = vec(var(Y; dims=2))

"""
    normalized_eval(model, inputs, Y, baseline; solver, reltol, abstol, dt) → Vector{Float64}

Per-component normalized error E/Var
"""
function normalized_eval(model, inputs::Vector, Y::AbstractMatrix,
    baseline::AbstractVector;
    solver=Tsit5(), reltol=1e-9, abstol=1e-12, dt=nothing)
    Yhat = batched_predict(model, inputs; solver, reltol, abstol, dt)
    return vec(sum(abs2, Yhat .- Y; dims=2)) ./ (size(Y, 2) .* baseline)
end

"""
    heldout_error(model, task, n; seed) → Float64

Per-sample error on a fresh evaluation batch
"""
function heldout_error(model, task::AbstractTask, n::Int; seed::Int)
    inputs, targets = generate(task, n; rng=Xoshiro(seed + EVAL_OFFSET))
    return compute_error(model, inputs, targets) / n
end

"""
    ConvergenceTracker(τ; every=200, consecutive=2, plateau_evals=0, min_improve=0.02)
"""
mutable struct ConvergenceTracker
    τ::Float64
    every::Int
    consecutive::Int
    plateau_evals::Int
    min_improve::Float64
    epochs::Vector{Int}
    values::Vector{Vector{Float64}}
    below::Int
    epoch_converged::Union{Int,Nothing}
    best::Float64
    evals_since_improve::Int
    epoch_plateaued::Union{Int,Nothing}
end

ConvergenceTracker(τ::Real; every::Int=200, consecutive::Int=2,
    plateau_evals::Int=0, min_improve::Real=0.02) =
    ConvergenceTracker(Float64(τ), every, consecutive, plateau_evals,
        Float64(min_improve), Int[], Vector{Float64}[], 0, nothing,
        Inf, 0, nothing)

converged(t::ConvergenceTracker) = t.epoch_converged !== nothing
plateaued(t::ConvergenceTracker) = t.epoch_plateaued !== nothing

"""
    final_values(t, d) → Vector{Float64}
    track_matrix(t, d) → Matrix{Float64}
"""
final_values(t::ConvergenceTracker, d::Int) =
    isempty(t.values) ? fill(NaN, d) : t.values[end]
track_matrix(t::ConvergenceTracker, d::Int) =
    isempty(t.values) ? zeros(d, 0) : reduce(hcat, t.values)

"""
    tracker_callback(t, eval_fn; metric=first, lg=nothing, label="") → Function
"""
function tracker_callback(t::ConvergenceTracker, eval_fn;
    metric=first, lg=nothing, label::AbstractString="")
    return (E, epoch, errors) -> begin
        epoch % t.every == 0 || return nothing
        comp = eval_fn()
        push!(t.epochs, epoch)
        push!(t.values, comp)
        m = metric(comp)
        logln(lg, @sprintf("%s epoch %6d  E/Var=%.4f", label, epoch, m))
        t.below = m < t.τ ? t.below + 1 : 0
        if t.below >= t.consecutive
            t.epoch_converged === nothing && (t.epoch_converged = epoch)
            return :stop
        end
        if m < t.best * (1 - t.min_improve)
            t.best = m
            t.evals_since_improve = 0
        else
            t.evals_since_improve += 1
        end
        if t.plateau_evals > 0 && t.evals_since_improve >= t.plateau_evals
            if t.epoch_plateaued === nothing
                t.epoch_plateaued = epoch
                logln(lg, @sprintf("%s epoch %6d  plateau stop (best E/Var=%.4f)",
                    label, epoch, t.best))
            end
            return :stop
        end
        return nothing
    end
end

"""
    train_run(task; hidden_dim, ncp, seed, n_epochs, num_samples, kwargs...)
    train_run(make_callback, task; kwargs...)
        → (model, errors, wall)

Shared experiment loop
"""
function train_run(make_callback::Function, task::AbstractTask;
    hidden_dim::Int=0, ncp::Int=0, seed::Int,
    pool::Union{Nothing,DataPool}=nothing, num_samples::Int,
    sigma=tanh, RepType::Type=Spline, T::Type=Float64, init_scale=0.3,
    n_epochs::Int, learning_rate=3e-3,
    patience::Int=n_epochs, lr_decay=1.0, min_lr=1e-8,
    solver=Tsit5(), reltol=1e-9, abstol=1e-12, dt=nothing, eval_every::Int=0,
    sensealg=_default_sensealg(), data_to_device=identity,
    backend::Symbol=:sciml, nsteps::Int=32,
    model=nothing, teacher=nothing,
    init_strategy::Union{Nothing,Function}=nothing,
    checkpoint::Union{Nothing,Tuple}=nothing, checkpoint_every::Int=0,
    resume_ckpt::Bool=resume_enabled())

    Random.seed!(seed)
    user_model = model
    if model === nothing
        model = init_model(task, hidden_dim, ncp;
            sigma=sigma, RepType=RepType, init_scale=Float64(init_scale), T=T)
    end
    cpath = checkpoint === nothing ? nothing : ckpt_path(checkpoint[1], checkpoint[2])
    resume_from = (cpath !== nothing && resume_ckpt && user_model === nothing && isfile(cpath)) ?
                  cpath : nothing
    on_epoch = make_callback(model)
    rng = Xoshiro(seed + DATA_OFFSET)
    generate_batch = pool === nothing ?
                     (() -> generate(task, num_samples; rng=rng)) :
                     generate_sampler(pool, num_samples; rng=rng)
    if teacher !== nothing
        generate_data = generate_batch
        generate_batch = () -> begin
            inputs, _ = generate_data()
            Yhat = batched_predict(teacher, inputs; solver, reltol, abstol, dt)
            (inputs, [Yhat[:, i] for i in 1:size(Yhat, 2)])
        end
    end

    if init_strategy !== nothing && user_model === nothing && resume_from === nothing
        in0, tgt0 = generate_batch()
        model = init_strategy(model, in0, tgt0)
    end

    t0 = time()
    _, errors = train_adjoint(model, generate_batch;
        learning_rate, n_epochs, verbose=0,
        num_samples, patience, lr_decay, min_lr,
        solver, reltol, abstol, dt, eval_every, sensealg, data_to_device,
        backend, nsteps, on_epoch,
        checkpoint_every, checkpoint_path=cpath, resume_from)
    wall = time() - t0
    return model, errors, wall
end

train_run(task::AbstractTask; kwargs...) = train_run(_ -> nothing, task; kwargs...)

"""
    tracked_train_run(task; eval_inputs, eval_targets, τ, metric=maximum,
                      every=200, consecutive=2, label="", lg=nothing, kwargs...)
        → (; model, errors, wall, tracker, baseline)

`train_run` with convergence tracking
"""
function tracked_train_run(task::AbstractTask; eval_inputs, eval_targets,
    τ::Float64, metric=maximum, every::Int=200, consecutive::Int=2,
    plateau_evals::Int=0, min_improve::Real=0.02,
    label::AbstractString="", lg=nothing,
    solver=Tsit5(), reltol=1e-9, abstol=1e-12, dt=nothing, kwargs...)
    Y = reduce(hcat, eval_targets)
    baseline = target_baseline(Y)
    tracker = ConvergenceTracker(τ; every, consecutive, plateau_evals, min_improve)
    model, errors, wall = train_run(task;
        solver, reltol, abstol, dt, kwargs...) do m
        tracker_callback(tracker,
            () -> normalized_eval(m, eval_inputs, Y, baseline;
                solver, reltol, abstol, dt);
            metric, lg, label)
    end
    return (; model, errors, wall, tracker, baseline)
end

end
