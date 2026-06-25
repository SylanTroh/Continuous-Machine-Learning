#!/usr/bin/env julia
# Usage:  julia --project=. experiments/plots.jl [plotname|all]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using NeuralFlow
using Plots
using Printf
using Statistics: mean

function _parse_log_losses(dir, arms)
    logs = filter(f -> endswith(f, ".log"), readdir(dir; join=true))
    isempty(logs) && error("no logs in $dir")
    rx = Regex("config_(\\d+)\\s+($(join(arms, "|")))\\s+epoch\\s+(\\d+)\\s+E=([-\\d.eE+]+)")
    seen = Dict(a => Dict{Tuple{Int,Int},Float64}() for a in arms)
    for lf in logs, ln in eachline(lf)
        m = match(rx, ln)
        m === nothing && continue
        seen[m.captures[2]][(parse(Int, m.captures[1]), parse(Int, m.captures[3]))] =
            parse(Float64, m.captures[4])
    end
    all(isempty, values(seen)) && error("no E= lines parsed from logs in $dir")
    return seen
end

_logloss_median(v) = (s = sort(v); n = length(s);
    isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function _logloss_curve(seen_arm, cfgs; agg=_logloss_median)
    byep = Dict{Int,Vector{Float64}}()
    for ((cfg, ep), v) in seen_arm
        (cfgs === nothing || cfg in cfgs) && push!(get!(byep, ep, Float64[]), v)
    end
    eps = sort(collect(keys(byep)))
    return eps, [agg(byep[e]) for e in eps]
end

const _SOLVER_NAME = Dict("Euler" => "Euler", "RK4" => "RK4",
    "Tsit5" => "Tsit5", "Tsit5adaptive" => "Tsit5 (adaptive)")
const _SOLVER_COLOR = Dict("Euler" => :indianred, "RK4" => :seagreen,
    "Tsit5" => :steelblue, "Tsit5adaptive" => :darkorange)

function _solver_loss_curve(samp)
    L = minimum(length.(samp))
    return [mean(c[k] for c in samp) for k in 1:L]
end

function plot_solver_loss(; dir=experiment_dir("paired_solver_test"))
    groups = [("det(3)", 1), ("MNIST", 2)]
    groups = filter(g -> isfile(joinpath(dir, "config_$(g[2]).jls")), groups)
    isempty(groups) && error("no config_*.jls in $dir")
    panels = map(enumerate(groups)) do (i, (tname, idx))
        f = joinpath(dir, "config_$(idx).jls")
        nt = load_run(f)
        eps = collect(nt.config.sample_epochs)
        p = plot(; xlabel="epoch", ylabel=(i == 1 ? "training loss" : ""),
            yscale=:log10, xformatter=:plain, legend=:topright, title=tname,
            titlefontsize=12, guidefontsize=10, tickfontsize=8, legendfontsize=9)
        for (j, be) in enumerate(nt.backends)
            ys = _solver_loss_curve(nt.samples[j])
            plot!(p, eps[1:length(ys)], ys; c=get(_SOLVER_COLOR, be, :black), lw=2,
                label=get(_SOLVER_NAME, be, be))
        end
        p
    end
    return plot(panels...; layout=(1, length(panels)), size=(820, 380),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
end

const _SR_KCOLOR = Dict(4 => :indianred, 8 => :darkorange, 16 => :seagreen,
    32 => :steelblue, 64 => :mediumorchid)

function plot_solver_resolution_loss(; dir=experiment_dir("solver_resolution"))
    KS = [4, 8, 16, 32, 64]
    families = [("Euler", "euler_steps"), ("RK4", "rk_steps")]
    panels = map(enumerate(families)) do (i, (fname, split))
        p = plot(; xlabel="epoch", ylabel=(i == 1 ? "training loss" : ""),
            yscale=:log10, xformatter=:plain, legend=:topright, title=fname,
            titlefontsize=12, guidefontsize=10, tickfontsize=8, legendfontsize=9)
        for k in KS
            f = joinpath(dir, "$(split)_k$(k).jls")
            isfile(f) || continue
            s = load_run(f).summary
            (hasproperty(s, :samples) && !isempty(s.samples)) || continue
            ys = _solver_loss_curve(s.samples)
            eps = collect(s.sample_epochs)[1:length(ys)]
            plot!(p, eps, ys; c=_SR_KCOLOR[k], lw=2, label="k=$k")
        end
        p
    end
    return plot(panels...; layout=(1, 2), size=(820, 380),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
end

function _escape_perseed(edir, base)
    cands = String[]
    main = joinpath(edir, base * ".model.jls")
    isfile(main) && push!(cands, main)
    archives = sort(filter(f -> startswith(f, base * ".model.") && f != base * ".model.jls" &&
                                endswith(f, ".jls"), readdir(edir)); rev=true)
    append!(cands, joinpath.(edir, archives))
    for p in cands
        m = load_run(p).models
        if hasproperty(m, :per_seed_curves) && maximum(length, m.per_seed_curves; init=0) > 0
            return m.per_seed_curves
        end
    end
    error("no per_seed_curves in any $base checkpoint under $edir")
end

function plot_det_plateau(; dir=experiment_dir("det_plateau"))
    EVAL_EVERY = 250
    NUM_SAMPLES = 128
    detvar(n) = mean(load_run(joinpath(dir, "det$(n).jls")).summary.vars)

    panels = NamedTuple[]
    let s = load_run(joinpath(dir, "det3.jls")).summary
        curves = [s.samples[i] ./ s.vars[i] for i in eachindex(s.samples)]
        push!(panels, (n=3, curves=curves, epochs=collect(s.sample_epochs)))
    end
    for n in (4, 5)
        psc = _escape_perseed(experiment_dir("det$(n)_escape"),
            "determinant_$(n)_N$(NUM_SAMPLES)_h64_ncp8")
        v = detvar(n)
        curves = [c ./ NUM_SAMPLES ./ v for c in psc]
        nck = maximum(length, curves; init=0)
        push!(panels, (n=n, curves=curves, epochs=collect(EVAL_EVERY:EVAL_EVERY:(EVAL_EVERY * nck))))
    end

    plts = map(panels) do pan
        p = plot(; xlabel="epoch", ylabel="E/Var", yscale=:log10,
            xformatter=:plain, legend=false, title="det($(pan.n))",
            titlefontsize=12, guidefontsize=10, tickfontsize=8)
        for c in pan.curves
            plot!(p, pan.epochs[1:length(c)], max.(c, 1e-6); c=:steelblue, lw=1, alpha=0.35)
        end
        L = minimum(length.(pan.curves); init=0)
        if L > 0
            med = [_logloss_median([c[k] for c in pan.curves]) for k in 1:L]
            plot!(p, pan.epochs[1:L], max.(med, 1e-6); c=:black, lw=2)
        end
        p
    end
    return plot(plts...; layout=(1, 3), size=(1320, 420),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
end

function plot_paired_loss(; dir=experiment_dir("paired_test"))
    arms = ("Spline", "ChebPoly")
    seen = _parse_log_losses(dir, arms)
    groups = [("det(3)", 1:4), ("MNIST", 5:8)]
    armcolor = Dict("Spline" => :steelblue, "ChebPoly" => :darkorange)
    armname = Dict("Spline" => "Spline", "ChebPoly" => "Chebyshev")
    panels = map(enumerate(groups)) do (i, (tname, cfgs))
        p = plot(; xlabel="epoch", ylabel=(i == 1 ? "training loss" : ""),
            yscale=:log10, xformatter=:plain, legend=:topright, title=tname,
            titlefontsize=12, guidefontsize=10, tickfontsize=8, legendfontsize=9)
        for a in arms
            eps, ys = _logloss_curve(seen[a], cfgs; agg=mean)
            isempty(eps) && continue
            plot!(p, eps, ys; c=armcolor[a], lw=2, label=armname[a])
        end
        p
    end
    return plot(panels...; layout=(1, length(panels)), size=(820, 380),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
end

# ── cell_depth: mean training-loss convergence per block arm (det3 + MNIST) ──
_is_archive(f) = occursin(r"\.\d{4}-\d{2}-\d{2}T\d{6}(?:-\d+)?\.jls$", f)

function _cell_depth_cells(; dir=experiment_dir("cell_depth"))
    out = NamedTuple[]
    isdir(dir) || return out
    for f in sort(readdir(dir))
        endswith(f, ".jls") && !_is_archive(f) || continue
        (endswith(f, ".model.jls") || endswith(f, ".ckpt.jls")) && continue
        nt = load_run(joinpath(dir, f))
        haskey(nt, :summary) && haskey(nt.summary, :spec) && push!(out, nt.summary)
    end
    return out
end

const _CD_SPEC_COLOR = Dict("L1" => :steelblue, "L2" => :seagreen, "bottleneck" => :darkorange)
const _CD_SPEC_NAME = Dict("L1" => "L1", "L2" => "L2", "bottleneck" => "bottleneck")
const _CD_SPEC_ORDER = Dict("L1" => 1, "L2" => 2, "bottleneck" => 3)

_cd_series(cells) = sort(unique((s.spec, s.nsteps) for s in cells);
    by=c -> (get(_CD_SPEC_ORDER, c[1], 9), -c[2]))

function _cd_label(spec, nst)
    name = _CD_SPEC_NAME[spec]
    nst == 16 && return name
    endswith(name, ")") ? "$(chop(name)), $(nst) steps)" : "$name ($(nst) steps)"
end

function plot_cell_depth_loss(; dir=experiment_dir("cell_depth"))
    allcells = _cell_depth_cells(; dir)
    isempty(allcells) && error("no cell_depth cells in $dir")
    tasks = [t for t in ("det3", "mnist") if any(c.task == t for c in allcells)]
    panels = map(enumerate(tasks)) do (i, task)
        cells = [c for c in allcells if c.task == task]
        p = plot(; xlabel="epoch", ylabel=(i == 1 ? "training loss" : ""),
            yscale=:log10, xformatter=:plain, legend=:topright, title=task,
            titlefontsize=12, guidefontsize=10, tickfontsize=8, legendfontsize=9)
        for (spec, nst) in _cd_series(cells)
            (spec == "L2" && nst == 16) && continue   # show only the compute-matched L2 (8 steps)
            arm = [c for c in cells if c.spec == spec && c.nsteps == nst]
            isempty(arm) && continue
            curves = [c.train_curve ./ (c.nseeds * c.num_samples) for c in arm]
            L = minimum(length, curves)
            ys = [sum(c[k] for c in curves) / length(curves) for k in 1:L]
            plot!(p, 1:L, ys; c=get(_CD_SPEC_COLOR, spec, :gray), lw=2,
                label=_cd_label(spec, nst))
        end
        p
    end
    return plot(panels...; layout=(1, length(panels)), size=(820, 380),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)
end

function _save(plt, file; out)
    mkpath(out)
    savefig(plt, joinpath(out, file))
    println("saved → $(joinpath(out, file))")
end

function save_figures(which::AbstractString="all";
    out=joinpath(@__DIR__, "..", "Figures"))
    attempt(f) = try
        f()
    catch e
        println("skipped: ", sprint(showerror, e))
    end
    if which in ("paired", "all")
        attempt(() -> _save(plot_paired_loss(), "paired_loss.png"; out))
    end
    if which in ("solverloss", "solver", "all")
        attempt(() -> _save(plot_solver_loss(), "solver_loss.png"; out))
    end
    if which in ("solverres", "solver", "all")
        attempt(() -> _save(plot_solver_resolution_loss(), "solver_resolution_loss.png"; out))
    end
    if which in ("detplateau", "plateau", "det", "all")
        attempt(() -> _save(plot_det_plateau(), "det_plateau.png"; out))
    end
    if which in ("celldepth", "cell", "all")
        attempt(() -> _save(plot_cell_depth_loss(), "cell_depth_loss.png"; out))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_figures(isempty(ARGS) ? "all" : ARGS[1])
end
