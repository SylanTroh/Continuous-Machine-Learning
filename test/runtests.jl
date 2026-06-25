#!/usr/bin/env julia
# Quick tests:  julia --project=. test/runtests.jl
# All tests:    julia --project=. test/runtests.jl --all

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using NeuralFlow
using NeuralFlow.GradientUtils
using DifferentialEquations: Tsit5
using Adapt: adapt
import JLArrays
import Zygote
using LinearAlgebra
using Random
import Plots

run_slow = "--all" in ARGS

#Julia threads × multithreaded OpenBLAS can spawn too many OpenBLAS threads
Threads.nthreads() > 1 && LinearAlgebra.BLAS.set_num_threads(1)

passed = 0
failed = 0
skipped = 0

#Add new function representations here.
const REP_TYPES = (Spline, ChebPoly)

function _run_case(name, body)
    print("  ", name, "... ")
    try
        body()
        println("OK")
        global passed += 1
    catch e
        println("FAIL")
        showerror(stdout, e, catch_backtrace())
        println()
        global failed += 1
    end
end

macro testcase(name, body)
    quote
        _run_case($name, () -> $(esc(body)))
    end
end

macro slowtest(name, body)
    quote
        if run_slow
            _run_case($name, () -> $(esc(body)))
        else
            print("  ", $name, "... SKIPPED (use --all)\n")
            global skipped += 1
        end
    end
end

relerr(g, ref) = norm(g .- ref) / max(1e-8, norm(ref))

small_model(h_dim, ncp; in_d=4, out_d=1, sigma=tanh, scale=0.3) =
    NeuralFlowODE(
        Spline([scale .* randn(h_dim, h_dim) for _ in 1:ncp]),
        Spline([scale .* randn(h_dim) for _ in 1:ncp]),
        scale .* randn(h_dim, in_d), scale .* randn(out_d, h_dim), sigma)

function _model_loss(model, input, target, solver_fn;
    reltol::Float64=1e-12, abstol::Float64=1e-14)
    a = model.W_in * vec(input)
    time_steps = collect(range(0.0, 1.0, length=4 * length(model.B)))
    _, xs = solver_fn(model, a, time_steps; reltol=reltol, abstol=abstol)
    r = model.W_out * xs[end] .- (target isa AbstractVector ? target : [target])
    return sum(abs2, r)
end

function _fd_grad(val, P; ε=1e-6)
    g = zero(P)
    for i in eachindex(P)
        Pp = copy(P)
        Pp[i] += ε
        Pm = copy(P)
        Pm[i] -= ε
        g[i] = (val(Pp) - val(Pm)) / (2ε)
    end
    return g
end

_flatP(g) = vcat(vec(g.A), vec(g.B), vec(g.W_in), vec(g.W_out))

println("Function Representation")

@testcase "f(t) ≈ dot(basis_weights(f,t), vals(f))" begin
    Random.seed!(3)
    for R in REP_TYPES
        F = random_like(R, 6)
        for t in (0.0, 0.137, 0.5, 0.731, 1.0)
            @assert isapprox(dot(basis_weights(F, t), vals(F)), F(t); atol=1e-10) "$(nameof(R)) at t=$t"
        end
    end
end

@testcase "basis_weights sum to 1" begin
    for R in REP_TYPES
        F = random_like(R, 7)
        for t in (0.1, 0.33, 0.5, 0.7, 0.9)
            @assert sum(basis_weights(F, t)) ≈ 1.0 atol = 1e-12
        end
    end
end

@testcase "arithmetic (+, -, scalar *)" begin
    Random.seed!(5)
    for R in REP_TYPES
        f = random_like(R, 4)
        g = random_like(R, 4)
        for t in (0.0, 0.3, 1.0)
            @assert (f + g)(t) ≈ f(t) + g(t) "$(nameof(R)) + at t=$t"
            @assert (f - g)(t) ≈ f(t) - g(t) "$(nameof(R)) - at t=$t"
            @assert (2.0 * f)(t) ≈ 2.0 * f(t) "$(nameof(R)) * at t=$t"
            @assert (-f)(t) ≈ -(f(t)) "$(nameof(R)) negate at t=$t"
        end
    end
end

@testcase "device_basis_weights matches basis_weights" begin
    Random.seed!(4)
    # don't index one at a time on the gpu
    JLArrays.allowscalar(false)
    x = adapt(JLArrays.JLArray, zeros(Float32, 3))
    for R in REP_TYPES
        F = random_like(R, 5)
        for t in (0.0, 0.13, 0.5, 0.999, 1.0)
            @assert device_basis_weights(F, t, zeros(3)) ≈ basis_weights(F, t) "$(nameof(R)) at t=$t"
        end
        w = device_basis_weights(F, 0.37, x)
        @assert w isa JLArrays.JLArray "$(nameof(R)) device weights"
        @assert adapt(Array, w) ≈ Float32.(basis_weights(F, 0.37)) "$(nameof(R)) device weights"
    end
end

@testcase "BasisCache memoizes device_basis_weights" begin
    Random.seed!(4)
    x = zeros(3)
    for R in REP_TYPES
        F = random_like(R, 5)
        c = BasisCache(F, x)
        for t in (0.0, 0.13, 0.5, 0.999, 1.0)
            @assert device_basis_weights(c, t, x) ≈ basis_weights(F, t) "$(nameof(R)) at t=$t"
        end
        @assert length(c.table) == 5 "$(nameof(R)) table size"
        w = device_basis_weights(c, 0.37, x)
        @assert device_basis_weights(c, 0.37, x) === w "$(nameof(R)) second lookup must hit"
    end
    JLArrays.allowscalar(false)
    xj = adapt(JLArrays.JLArray, zeros(Float32, 3))
    F = random_like(Spline, 5)
    c = BasisCache(F, xj)
    w = device_basis_weights(c, 0.37, xj)
    @assert w isa JLArrays.JLArray "cached weights live on the device"
    @assert adapt(Array, w) ≈ Float32.(basis_weights(F, 0.37))
end

@testcase "polynomial evaluation" begin
    affine(t) = 2.0 + 3.0 * t
    for R in REP_TYPES
        F0 = random_like(R, 5)
        F = reconstruct(F0, affine.(nodes(F0)))
        for t in (0.0, 0.13, 0.42, 0.78, 1.0)
            @assert abs(F(t) - affine(t)) < 1e-12 "$(nameof(R)) at t=$t"
        end
    end
    quad(t) = 2.0 + 3.0 * t - 4.0 * t^2
    G0 = random_like(ChebPoly, 5)
    G = reconstruct(G0, quad.(nodes(G0)))
    for t in (0.0, 0.13, 0.42, 0.78, 1.0)
        @assert abs(G(t) - quad(t)) < 1e-12 "ChebPoly quadratic at t=$t"
    end
end

@testcase "sine evaluation" begin
    target(t) = sin(2π * t)
    for R in REP_TYPES
        F0 = random_like(R, 33)
        F = reconstruct(F0, target.(nodes(F0)))
        for t in (0.0, 0.13, 0.42, 0.78, 1.0)
            @assert abs(F(t) - target(t)) < 1e-2 "$(nameof(R)) at t=$t"
        end
    end
end

@testcase "change_representation" begin
    # interpolation property: the new representation matches the original
    # exactly at its own nodes, for scalar-, vector-, and matrix-valued reps
    Random.seed!(1)
    A = random_like(Spline, 4, (3, 3))
    Ac = change_representation(A, ChebPoly, 9)
    @assert Ac isa ChebPoly
    for t in nodes(Ac)
        @assert maximum(abs.(Ac(t) .- A(t))) < 1e-12
    end
    # affine functions are exactly representable in every basis
    affine(t) = 2.0 + 3.0 * t
    F0 = random_like(Spline, 2)
    F = reconstruct(F0, affine.(nodes(F0)))
    G = change_representation(F, ChebPoly, 5)
    for t in (0.0, 0.13, 0.42, 0.78, 1.0)
        @assert abs(G(t) - affine(t)) < 1e-12
    end
    # model-level: control parameters re-represented, embeddings and σ kept
    B = random_like(Spline, 4, (3,))
    m = NeuralFlowODE(A, B; sigma=tanh)
    mc = change_representation(m, ChebPoly, 9)
    @assert mc.A isa ChebPoly && mc.B isa ChebPoly
    @assert mc.W_in === m.W_in && mc.W_out === m.W_out && mc.sigma === m.sigma
    a = randn(3)
    _, xs = solve_tsit5(m, a, 0:0.1:1)
    _, xc = solve_tsit5(mc, a, 0:0.1:1)
    @assert maximum(maximum(abs.(x .- y)) for (x, y) in zip(xs, xc)) < 0.2
end

println("\nLinearInterp")

@testcase "interpolate_samples" begin
    @assert interpolate_samples([0.0, 1.0], 0.5) ≈ 0.5
    @assert interpolate_samples([1.0, 2.0, 4.0], 0.75) ≈ 3.0
    #vector-valued samples, for solve_picard
    xs = [[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]
    @assert interpolate_samples(xs, 0.25) ≈ [2.0, 1.0]
    @assert interpolate_samples(xs, 1.0) ≈ [5.0, 4.0]
end

println("\nChebyshevRepresentation")

@testcase "Chebyshev norm" begin
    N = 16
    cheb_nodes = [0.5 * (1 - cos(π * k / (N - 1))) for k in 0:(N-1)]
    q = ChebPoly(cos.(π .* cheb_nodes))
    @assert abs(norm(q) - sqrt(0.5)) < 1e-10
end

@testcase "reconstruct preserves cached metadata" begin
    f = ChebPoly([1.0, 2.0, 3.0])
    g = reconstruct(f, [10.0, 20.0, 30.0])
    @assert g.nodes === f.nodes
    @assert g.bary === f.bary
    @assert g.basis_integrals === f.basis_integrals
end

println("\nSolvers")

@testcase "Tsit5 and Picard agree" begin
    tgrid = collect(range(0.0, 1.0, length=50))
    for R in REP_TYPES
        model = NeuralFlowODE(random_like(R, 5, (3, 3)), random_like(R, 5, (3,)); sigma=sigmoid)
        a = randn(3)
        @assert length(f(model, a, 0.5)) == 3 "$(nameof(R))"
        _, xs_e = solve_euler(model, a, tgrid)
        @assert length(xs_e) == 50 && xs_e[1] ≈ a "$(nameof(R)) euler"
        _, xs_tsit = solve_tsit5(model, a, tgrid; reltol=1e-12, abstol=1e-14)
        _, xs_pic = solve_picard(model, a, tgrid; maxit=30, atol=1e-12)
        max_err = maximum(norm(xs_tsit[i] - xs_pic[i]) for i in eachindex(tgrid))
        @assert max_err < 1e-4 "$(nameof(R)) Tsit5 vs Picard mismatch: $max_err"
    end
end

@testcase "forward_gradients" begin
    tgrid = collect(range(0.0, 1.0, length=50))
    for R in REP_TYPES
        model = NeuralFlowODE(random_like(R, 5, (3, 3)), random_like(R, 5, (3,)); sigma=sigmoid)
        gA, gB, gWi, gWo = forward_gradients(model, [randn(3) for _ in 1:2],
            [randn(3) for _ in 1:2], tgrid)
        @assert length(gA) == 5 && length(gB) == 5 "$(nameof(R))"
        @assert size(gWi) == size(model.W_in) "$(nameof(R))"
        @assert size(gWo) == size(model.W_out) "$(nameof(R))"
    end
end

println("\nGradientUtils")

@testcase "AdamFlat" begin
    Random.seed!(22)
    task = DeterminantTask(2)
    m = init_model(task, 6, 3; sigma=tanh, RepType=Spline, T=Float32)
    inputs, targets = generate(task, 8)
    Ain = Float32.(reduce(hcat, vec.(inputs)))
    Y = Float32.(reduce(hcat, targets))
    p = param_vector(m)
    opt = AdamFlat(p)
    E0, _ = batched_value_and_gradient(p, m, Ain, Y; reltol=1e-4, abstol=1e-6)
    for _ in 1:6
        _, g = batched_value_and_gradient(p, m, Ain, Y; reltol=1e-4, abstol=1e-6)
        apply_gradients!(opt, p, 1e-2, g)
    end
    E1, _ = batched_value_and_gradient(p, m, Ain, Y; reltol=1e-4, abstol=1e-6)
    @assert E1 < E0 "AdamFlat did not decrease loss: $E0 -> $E1"
end

println("\nExperimentUtils")

@testcase "run_parallel" begin
    active = Threads.Atomic{Int}(0)
    peak = Threads.Atomic{Int}(0)
    thunks = [() -> begin
        n = Threads.atomic_add!(active, 1) + 1
        Threads.atomic_max!(peak, n)
        sleep(0.02)
        Threads.atomic_sub!(active, 1)
        i
    end for i in 1:6]
    res = run_parallel(thunks; max_concurrent=2)
    @assert res == collect(1:6)
    @assert peak[] <= 2
end

@testcase "run_cell runs once, then loads the saved result" begin
    path = joinpath(mktempdir(), "cell.jls")
    calls = Ref(0)
    r1 = run_cell(() -> (calls[] += 1; (value=42,)), path)
    r2 = run_cell(() -> (calls[] += 1; (value=0,)), path)
    @assert calls[] == 1
    @assert r1.value == 42 && r2.value == 42
end

@testcase "paired helpers: seed_cell, num_concurrent, paired_ttest" begin
    @assert endswith(seed_cell("paired_test", "config_1", 7),
        joinpath("config_1_seeds", "seed_7.jls"))
    withenv("NUM_CONCURRENT" => nothing) do
        @assert num_concurrent(5) == 5
    end
    withenv("NUM_CONCURRENT" => "3") do
        @assert num_concurrent(5) == 3
    end
    d = [1.0, 2.0, 3.0, 4.0, 5.0]   # mean 3, var 2.5, n 5
    r = paired_ttest("demo", d; diff_label="A - B")
    @assert r.n == 5 && r.df == 4
    @assert r.mean == 3.0
    @assert isapprox(r.t, 3.0 / sqrt(2.5 / 5); atol=1e-9)
    @assert isapprox(r.cohen_dz, 3.0 / sqrt(2.5); atol=1e-9)
    @assert 0 < r.p_t < 0.05
    @assert r.diffs === d
    @assert r.ci[1] < r.mean < r.ci[2]
end

@testcase "TeeLog writes timestamped lines to the log file" begin
    path = joinpath(mktempdir(), "run.log")
    lg = TeeLog(path)
    logln(lg, "tee check")
    close(lg)
    line = readlines(path)[end]
    @assert occursin("tee check", line)
    @assert occursin(r"^\[\d\d:\d\d:\d\d\] ", line)
end

@testcase "target_baseline and normalized_eval" begin
    Random.seed!(7)
    t = DeterminantTask(2)
    m = init_model(t, 4, 3; sigma=tanh, RepType=Spline)
    ins, tg = generate(t, 16)
    base = target_baseline(tg)
    @assert length(base) == 1 && all(base .> 0)
    comp = normalized_eval(m, ins, reduce(hcat, tg), base; reltol=1e-4, abstol=1e-6)
    @assert length(comp) == 1 && all(isfinite, comp)
end

@testcase "tracked_train_run stops at convergence" begin
    t = DeterminantTask(2)
    ins, tg = generate(t, 8; rng=Xoshiro(1))
    r = tracked_train_run(t; eval_inputs=ins, eval_targets=tg,
        τ=1e9, every=2, hidden_dim=4, ncp=3, seed=11,
        num_samples=4, n_epochs=50, reltol=1e-4, abstol=1e-6)
    @assert converged(r.tracker) && r.tracker.epoch_converged == 4
    @assert length(r.errors) == 4
    @assert size(track_matrix(r.tracker, 1)) == (1, 2)
    @assert final_values(r.tracker, 1) == r.tracker.values[end]
    @assert r.wall > 0
end

@testcase "train_run draws from a fixed pool when given" begin
    t = DeterminantTask(2)
    pool = build_data_pool(t, 6; rng=Xoshiro(2))
    m, errs, _ = train_run(t; hidden_dim=4, ncp=3, seed=11, pool=pool,
        num_samples=4, n_epochs=3, reltol=1e-4, abstol=1e-6)
    @assert length(errs) == 3 && all(isfinite, errs)
    @assert isfinite(heldout_error(m, t, 8; seed=11))
end

@testcase "plateau stop" begin
    # constant metric: never converges, plateaus after plateau_evals evaluations
    t = ConvergenceTracker(1e-9; every=1, plateau_evals=3)
    cb = tracker_callback(t, () -> [0.5]; metric=first)
    stops = [cb(0.0, epoch, Float64[]) for epoch in 1:10]
    @assert !converged(t)
    @assert plateaued(t)
    @assert t.epoch_plateaued == 4 "plateaued at $(t.epoch_plateaued), expected 4"
    @assert stops[4] === :stop
    # steadily falling metric: no plateau
    vals = [1.0]
    t2 = ConvergenceTracker(1e-9; every=1, plateau_evals=3)
    cb2 = tracker_callback(t2, () -> (vals[1] *= 0.9; [vals[1]]); metric=first)
    foreach(epoch -> cb2(0.0, epoch, Float64[]), 1:20)
    @assert !plateaued(t2)
end

@testcase "train_run distillation path" begin
    Random.seed!(11)
    task = DeterminantTask(2)
    teacher = init_model(task, 8, 4; sigma=tanh, RepType=Spline, init_scale=0.3)
    student = change_representation(teacher, ChebPoly, 3)
    model, errors, _ = train_run(task;
        model=student, teacher=teacher, seed=11, num_samples=8, n_epochs=3,
        learning_rate=1e-3)
    @assert model === student
    @assert length(errors) == 3
    # the student should be near the teacher already (it starts as its interpolant)
    inputs, _ = generate(task, 16; rng=Xoshiro(99))
    Yt = batched_predict(teacher, inputs)
    Ys = batched_predict(model, inputs)
    @assert maximum(abs.(Yt .- Ys)) < 0.5
end

println("\nTasks")

@testcase "NestedDeterminantTask" begin
    t = NestedDeterminantTask(4)
    @assert input_dim(t) == 16
    @assert output_dim(t) == 4
    ins, outs = generate(t, 3)
    @assert length(ins) == 3 && length(outs) == 3
    for (M, tgt) in zip(ins, outs)
        @assert length(tgt) == 4
        for k in 1:4
            @assert tgt[k] ≈ det(M[k:4, k:4]) / sqrt(factorial(4 - k + 1))
        end
        @assert tgt[1] ≈ det(M) / sqrt(24)
        @assert tgt[4] ≈ M[4, 4]
    end
end

@testcase "CofactorDeterminantTask" begin
    t = CofactorDeterminantTask(4)
    @assert input_dim(t) == 16
    @assert output_dim(t) == 10
    ins, outs = generate(t, 2)
    for (M, tgt) in zip(ins, outs)
        @assert length(tgt) == 10
        @assert tgt[1] ≈ det(M) / sqrt(factorial(4))
        @assert det(M) ≈ sum(M[1, j] * sqrt(factorial(3)) * tgt[1+j] for j in 1:4)
        @assert tgt[2] ≈ det(M[2:4, 2:4]) / sqrt(factorial(3))
        B = M[2:4, 2:4]
        @assert det(B) ≈ sum(B[1, j] * sqrt(factorial(2)) * tgt[5+j] for j in 1:3)
        @assert tgt[9] ≈ M[4, 4]
        @assert tgt[10] ≈ -M[4, 3]
    end
end

@testcase "RandomMinorDeterminantTask" begin
    t = RandomMinorDeterminantTask(4; m=2, seed=7)
    t2 = RandomMinorDeterminantTask(4; m=2, seed=7)
    @assert output_dim(t) == 8
    ins, outs = generate(t, 2)
    nested = NestedDeterminantTask(4)
    for (M, tgt) in zip(ins, outs)
        @assert tgt[1:4] ≈ compute_target(nested, M)
        @assert compute_target(t2, M) ≈ tgt
        for (i, (rs, cs)) in enumerate(zip(t.rows, t.cols))
            @assert tgt[4+i] ≈ det(M[rs, cs]) / sqrt(factorial(length(rs)))
        end
    end
end

@testcase "has_target_fn" begin
    @assert has_target_fn(DeterminantTask(2))
end

println("\nFixedTrainingData")

@testcase "build_data_pool and sample_from_pool" begin
    Random.seed!(42)
    task = DeterminantTask(2)
    pool = build_data_pool(task, 8)
    @assert length(pool) == 8
    ins, outs = sample_from_pool(pool, 4)
    @assert length(ins) == 4 && length(outs) == 4
    @assert all(M in pool.inputs for M in ins)
end

@testcase "generate_sampler is deterministic" begin
    task = DeterminantTask(2)
    pool = build_data_pool(task, 8; rng=MersenneTwister(1))
    g1 = generate_sampler(pool, 3; rng=MersenneTwister(7))
    g2 = generate_sampler(pool, 3; rng=MersenneTwister(7))
    a, _ = g1()
    b, _ = g2()
    @assert a == b
end

println("\nDiscreteNN")

@testcase "forward is the Euler discretization of the continuous model" begin
    Random.seed!(7)
    d, K = 4, 5
    A = random_like(Spline, 6, (d, d))
    B = random_like(Spline, 6, (d,))
    cm = NeuralFlowODE(A, B; sigma=tanh)
    disc = DiscreteNetwork(d, d, K, d; sigma=tanh)
    for k in 1:K
        disc.As[k] = A((k - 1) / K)
        disc.Bs[k] = B((k - 1) / K)
    end
    disc.W_in = Matrix{Float64}(I, d, d)
    disc.W_out = Matrix{Float64}(I, d, d)
    a = randn(d)
    _, xs = solve_euler(cm, a, range(0.0, 1.0, length=K + 1))
    _, _, out = forward(disc, a)
    @assert maximum(abs.(out .- xs[end])) < 1e-12 "DiscreteNetwork ≠ Euler solve"
end

@testcase "backprop matches finite differences" begin
    Random.seed!(42)
    model = DiscreteNetwork(3, 4, 3, 2; sigma=tanh)
    inputs = [randn(3) for _ in 1:2]
    targets = [randn(2) for _ in 1:2]

    function E(m)
        s = 0.0
        for n in eachindex(inputs)
            _, _, output = forward(m, inputs[n])
            s += sum(abs2, output .- targets[n])
        end
        return s
    end

    gAs, gBs, gWi, gWo = backprop(model, inputs, targets)
    h = 1e-6
    function fd_check(P, g, name)
        max_err = 0.0
        for idx in eachindex(P)
            orig = P[idx]
            P[idx] = orig + h
            ep = E(model)
            P[idx] = orig - h
            em = E(model)
            P[idx] = orig
            fd = (ep - em) / (2h)
            err = abs(fd - g[idx]) / max(1.0, abs(fd))
            max_err = max(max_err, err)
        end
        @assert max_err < 1e-5 "$name finite difference mismatch: $max_err"
    end
    for k in eachindex(model.As)
        fd_check(model.As[k], gAs[k], "A_$k")
    end
    for k in eachindex(model.Bs)
        fd_check(model.Bs[k], gBs[k], "B_$k")
    end
    fd_check(model.W_in, gWi, "W_in")
    fd_check(model.W_out, gWo, "W_out")
end

@slowtest "DiscreteNN trains on DeterminantTask(2)" begin
    Random.seed!(123)
    t = DeterminantTask(2)
    model = DiscreteNetwork(input_dim(t), 16, 4, output_dim(t); sigma=tanh, init_scale=0.3)
    generate_batch() = generate(t, 32)
    # the residual form scales gradients by h = 1/K, so the learning rate is
    # higher than the continuous defaults
    _, errs = train_discrete(model, generate_batch;
        learning_rate=1e-2, n_epochs=100, verbose=0, num_samples=32,
        patience=2000, lr_decay=0.5, min_lr=1e-6)
    @assert minimum(errs) < errs[1] * 0.85 "DiscreteNN did not learn: $(errs[1]) -> $(minimum(errs))"
end

println("\nDerivatives")

let d = 3,
    model = NeuralFlowODE(RandomSpline(3, (3, 3)), RandomSpline(3, (3,)); sigma=relu),
    a = randn(3),
    tgrid = collect(range(0.0, 1.0, length=100)),
    traj = solve_euler(model, a, tgrid)[2]

    @testcase "D₁f matches finite difference" begin
        t = 0.3
        x = traj[30]
        J = D₁f(model, x, t)
        @assert size(J) == (d, d)
        f0 = f(model, x, t)
        h = 1e-7
        for j in 1:d
            ej = (e = zeros(d); e[j] = 1.0; e)
            fd = (f(model, x .+ h .* ej, t) .- f0) ./ h
            @assert norm(J[:, j] - fd) < 1e-4 "D₁f finite difference mismatch col $j: $(norm(J[:, j] - fd))"
        end
    end

    @testcase "D₂f matches finite difference" begin
        t = 0.3
        x = traj[30]
        h = 1e-7
        f0 = f(model, x, t)

        dA_vals = [zeros(size(model.A[k])) for k in 1:model.A.N]
        dA_vals[1][1, 1] = 1.0
        sym = D₂f(model, x, t, Spline(dA_vals)(t),
            Spline([zeros(size(model.B[k])) for k in 1:model.B.N])(t))

        A_pert = [copy(model.A[k]) for k in 1:model.A.N]
        A_pert[1][1, 1] += h
        fd = (f(NeuralFlowODE(Spline(A_pert), model.B; sigma=relu), x, t) .- f0) ./ h

        @assert norm(sym - fd) < 1e-4 "D₂f finite difference mismatch: $(norm(sym - fd))"
    end

    @testcase "solve_DPx matches finite difference" begin
        h = 1e-6

        dB_vals = [zeros(size(model.B[k])) for k in 1:model.B.N]
        dB_vals[1][1] = 1.0
        v = solve_DPx(model, traj, tgrid,
            Spline([zeros(size(model.A[k])) for k in 1:model.A.N]),
            Spline(dB_vals))
        @assert length(v) == length(tgrid) && v[1] ≈ zeros(d)

        B_pert = [copy(model.B[k]) for k in 1:model.B.N]
        B_pert[1][1] += h
        _, x_pert = solve_euler(
            NeuralFlowODE(Spline([copy(model.A[k]) for k in 1:model.A.N]),
                Spline(B_pert); sigma=relu),
            a, tgrid)

        fd = (x_pert[end] .- traj[end]) ./ h
        rel_err = norm(v[end] - fd) / max(norm(fd), 1e-10)
        @assert rel_err < 0.05 "DPx finite difference mismatch: $rel_err"
    end
end

println("\nForward Gradients")

@testcase "forward_gradients_tsit5 matches finite difference" begin
    Random.seed!(7)
    model = small_model(4, 4)
    inputs = [randn(2, 2) for _ in 1:2]
    targets = [[det(M)] for M in inputs]

    gA, gB, gWi, gWo = forward_gradients_tsit5(model, inputs, targets;
        reltol=1e-12, abstol=1e-14)
    fA, fB, fWi, fWo = fd_gradients(model, inputs, targets)

    for k in eachindex(fA)
        @assert relerr(gA[k], fA[k]) < 1e-3 "A_$k finite difference mismatch: $(relerr(gA[k], fA[k]))"
        @assert relerr(gB[k], fB[k]) < 1e-3 "B_$k finite difference mismatch: $(relerr(gB[k], fB[k]))"
    end
    @assert relerr(gWi, fWi) < 1e-3 "W_in finite difference mismatch: $(relerr(gWi, fWi))"
    @assert relerr(gWo, fWo) < 1e-3 "W_out finite difference mismatch: $(relerr(gWo, fWo))"
end

@testcase "forward_gradients (euler) matches finite difference" begin
    Random.seed!(17)
    model = small_model(3, 3)
    inputs = [randn(2, 2) for _ in 1:2]
    targets = [[det(M)] for M in inputs]
    tgrid = collect(range(0.0, 1.0, length=4 * 3))
    gA, gB, gWi, gWo = forward_gradients(model, inputs, targets, tgrid)
    @assert length(gA) == 3 && length(gB) == 3
    @assert size(gWi) == size(model.W_in)
    @assert size(gWo) == size(model.W_out)
    loss_euler() = sum(
        sum(abs2, model.W_out * last(solve_euler(model, model.W_in * vec(inputs[n]), tgrid)[2]) .- targets[n])
        for n in eachindex(inputs))
    h = 1e-5
    fd_Wo = zeros(size(model.W_out))
    for idx in eachindex(model.W_out)
        orig = model.W_out[idx]
        model.W_out[idx] = orig + h
        ep = loss_euler()
        model.W_out[idx] = orig - h
        em = loss_euler()
        model.W_out[idx] = orig
        fd_Wo[idx] = (ep - em) / (2h)
    end
    @assert relerr(gWo, fd_Wo) < 1e-3 "W_out finite difference mismatch: $(relerr(gWo, fd_Wo))"
end

println("\nResults & Checkpointing")

@testcase "save_run / load_run" begin
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "run.jls")
    A_pts = [randn(3, 3) for _ in 1:3]
    B_pts = [randn(3) for _ in 1:3]
    model = NeuralFlowODE(Spline(A_pts), Spline(B_pts),
        randn(3, 3), randn(1, 3), relu)
    errs = [1.0, 0.5, 0.2]
    save_run(path; model=model, errors=errs, config=(method=:test, ncp=3))
    r = load_run(path)
    @assert r.errors == errs
    @assert r.config.method === :test
    @assert size(r.model.W_in) == (3, 3)
end

@testcase "Checkpointer callbacks" begin
    tmpdir = mktempdir()
    path = joinpath(tmpdir, "cp.jls")
    cp = Checkpointer(path, 2; metadata=(test=true,))
    A_pts = [randn(2, 2) for _ in 1:2]
    B_pts = [randn(2) for _ in 1:2]
    model = NeuralFlowODE(Spline(A_pts), Spline(B_pts),
        randn(2, 2), randn(1, 2), relu)
    on_improve!(cp, 5.0, 1, copy_model(model))
    checkpoint!(cp, 2, [10.0, 5.0])
    @assert isfile(path)
    r = load_run(path)
    @assert r.best_error == 5.0
    @assert r.epoch == 2
end

println("\nAdjoint")

@testcase "adjoint_gradients matches finite difference" begin
    Random.seed!(7)
    model = small_model(4, 4)
    inputs = [randn(2, 2) for _ in 1:2]
    targets = [[det(M)] for M in inputs]

    gA, gB, gWi, gWo = adjoint_gradients(model, inputs, targets;
        reltol=1e-12, abstol=1e-14)
    fA, fB, fWi, fWo = fd_gradients(model, inputs, targets)

    for k in eachindex(fA)
        @assert relerr(gA[k], fA[k]) < 1e-3 "A_$k finite difference mismatch: $(relerr(gA[k], fA[k]))"
        @assert relerr(gB[k], fB[k]) < 1e-3 "B_$k finite difference mismatch: $(relerr(gB[k], fB[k]))"
    end
    @assert relerr(gWi, fWi) < 1e-3 "W_in finite difference mismatch: $(relerr(gWi, fWi))"
    @assert relerr(gWo, fWo) < 1e-3 "W_out finite difference mismatch: $(relerr(gWo, fWo))"
end

@testcase "Float32 gradients match Float64" begin
    Random.seed!(21)
    task = DeterminantTask(2)
    m32 = init_model(task, 6, 3; sigma=tanh, RepType=Spline, init_scale=0.3, T=Float32)
    @assert eltype(param_vector(m32)) == Float32
    m64 = NeuralFlowODE(
        Spline([Float64.(Ak) for Ak in vals(m32.A)]),
        Spline([Float64.(Bk) for Bk in vals(m32.B)]),
        Float64.(m32.W_in), Float64.(m32.W_out), m32.sigma)
    inputs, targets = generate(task, 4)
    E32, gA32, gB32, gWi32, gWo32 = adjoint_value_and_gradients(m32, inputs, targets;
        reltol=1e-4, abstol=1e-6)
    @assert E32 isa Float32
    @assert eltype(gA32[1]) == Float32 && eltype(gWo32) == Float32
    gA64, gB64, gWi64, gWo64 = adjoint_gradients(m64, inputs, targets;
        reltol=1e-9, abstol=1e-12)
    for k in eachindex(gA64)
        @assert relerr(gA32[k], gA64[k]) < 1e-2 "A_$k Float32 mismatch: $(relerr(gA32[k], gA64[k]))"
        @assert relerr(gB32[k], gB64[k]) < 1e-2 "B_$k Float32 mismatch: $(relerr(gB32[k], gB64[k]))"
    end
    @assert relerr(gWi32, gWi64) < 1e-2 "W_in Float32 mismatch: $(relerr(gWi32, gWi64))"
    @assert relerr(gWo32, gWo64) < 1e-2 "W_out Float32 mismatch: $(relerr(gWo32, gWo64))"
end

@testcase "fixed-step gradient matches adaptive" begin
    Random.seed!(23)
    task = DeterminantTask(2)
    m = init_model(task, 6, 3; sigma=tanh, RepType=Spline, init_scale=0.3)
    inputs, targets = generate(task, 4)
    Ain = reduce(hcat, vec.(inputs))
    Y = reduce(hcat, targets)
    p = param_vector(m)
    E_a, g_a = batched_value_and_gradient(p, m, Ain, Y; reltol=1e-9, abstol=1e-12)
    E_f, g_f = batched_value_and_gradient(p, m, Ain, Y; dt=1 / 64)
    @assert abs(E_f - E_a) / E_a < 1e-4 "E fixed vs adaptive mismatch: $(abs(E_f - E_a) / E_a)"
    @assert norm(g_f .- g_a) / norm(g_a) < 1e-3 "gradient fixed vs adaptive mismatch: $(norm(g_f .- g_a) / norm(g_a))"
end

_f_flat_ref(X, p, t, σ, repA, repB, d, nA) = begin
    wA = device_basis_weights(repA, t, X)
    wB = device_basis_weights(repB, t, X)
    At = reshape(reshape(p.A, d * d, nA) * wA, d, d)
    Bt = p.B * wB
    σ.(At * X .+ Bt)
end

@testcase "_f_flat matches Zygote" begin
    Random.seed!(41)
    softplus(z) = log1p(exp(z))
    task = DeterminantTask(2)
    d, nA = 5, 3
    for σ in (tanh, sigmoid, softplus), R in (Spline, ChebPoly)
        m = init_model(task, d, nA; sigma=σ, RepType=R)
        p = param_vector(m)
        X = randn(d, 7)
        for t in (0.0, 0.4, 1.0)
            gX_rule, gp_rule = Zygote.gradient(
                (Xv, q) -> sum(abs2, NeuralFlow.Adjoint._f_flat(Xv, q, t, σ, m.A, m.B, d, nA)), X, p)
            gX_ref, gp_ref = Zygote.gradient(
                (Xv, q) -> sum(abs2, _f_flat_ref(Xv, q, t, σ, m.A, m.B, d, nA)), X, p)
            @assert relerr(gX_rule, gX_ref) < 1e-9 "$(Symbol(σ)) $(nameof(R)) X-gradient at t=$t: $(relerr(gX_rule, gX_ref))"
            @assert relerr(collect(gp_rule), collect(gp_ref)) < 1e-9 "$(Symbol(σ)) $(nameof(R)) p-gradient at t=$t: $(relerr(collect(gp_rule), collect(gp_ref)))"
        end
    end
end

@testcase "_f_flat rrule on JLArray" begin
    JLArrays.allowscalar(false)
    Random.seed!(43)
    task = DeterminantTask(2)
    d, nA = 4, 3
    m = init_model(task, d, nA; sigma=tanh, RepType=Spline, T=Float32)
    p = param_vector(m)
    X = randn(Float32, d, 6)
    t = 0.4f0
    loss(Xv, q, rA, rB) = sum(abs2, NeuralFlow.Adjoint._f_flat(Xv, q, t, tanh, rA, rB, d, nA))
    gX_cpu, gp_cpu = Zygote.gradient((Xv, q) -> loss(Xv, q, m.A, m.B), X, p)
    pj = adapt(JLArrays.JLArray, p)
    Xj = adapt(JLArrays.JLArray, X)
    gX_j, gp_j = Zygote.gradient((Xv, q) -> loss(Xv, q, m.A, m.B), Xj, pj)
    @assert relerr(adapt(Array, gX_j), gX_cpu) < 1e-5 "X-gradient JLArray vs CPU"
    @assert relerr(collect(adapt(Array, gp_j)), collect(gp_cpu)) < 1e-5 "p-gradient JLArray vs CPU"
end

@testcase "batched_value_and_gradient with basis_caches matches without" begin
    Random.seed!(23)
    task = DeterminantTask(2)
    m = init_model(task, 4, 3; sigma=tanh, RepType=Spline)
    inputs, targets = generate(task, 4)
    Ain = reduce(hcat, vec.(inputs))
    Y = reduce(hcat, targets)
    p = param_vector(m)
    E0, g0 = batched_value_and_gradient(p, m, Ain, Y; dt=1 / 32)
    caches = (BasisCache(m.A, Ain), BasisCache(m.B, Ain))
    E1, g1 = batched_value_and_gradient(p, m, Ain, Y; dt=1 / 32, basis_caches=caches)
    @assert isapprox(E1, E0; rtol=1e-10) "E cached vs direct: $E1 vs $E0"
    @assert relerr(g1, g0) < 1e-10 "gradient cached vs direct: $(relerr(g1, g0))"
    @assert !isempty(caches[1].table) && !isempty(caches[2].table)
    E2, g2 = batched_value_and_gradient(p, m, Ain, Y; dt=1 / 32, basis_caches=caches)
    @assert isapprox(E2, E0; rtol=1e-10) && relerr(g2, g0) < 1e-10
end

@testcase "train_adjoint eval_every=0 reports training-batch errors" begin
    Random.seed!(31)
    t = DeterminantTask(2)
    m = init_model(t, 6, 3; sigma=tanh, RepType=Spline)
    gb() = generate(t, 4)
    _, errs = train_adjoint(m, gb; learning_rate=1e-3, n_epochs=5, verbose=0,
        num_samples=4, solver=Tsit5(), reltol=1e-4, abstol=1e-6, eval_every=0)
    @assert length(errs) == 5
    @assert all(isfinite, errs)
end

@testcase "train_adjoint stops when on_epoch returns :stop" begin
    Random.seed!(31)
    t = DeterminantTask(2)
    m = init_model(t, 6, 3; sigma=tanh, RepType=Spline)
    gb() = generate(t, 4)
    _, errs = train_adjoint(m, gb; learning_rate=1e-3, n_epochs=50, verbose=0,
        num_samples=4, solver=Tsit5(), reltol=1e-4, abstol=1e-6,
        on_epoch=(E, ep, errors) -> ep >= 3 ? :stop : nothing)
    @assert length(errs) == 3
end

@slowtest "train_adjoint trains on DeterminantTask" begin
    Random.seed!(99)
    t = DeterminantTask(2)
    tmpdir = mktempdir()
    cp_path = joinpath(tmpdir, "resume.jls")

    model = init_model(t, 8, 4; sigma=tanh, RepType=Spline, init_scale=0.3)
    generate_batch() = generate(t, 64)
    n_callbacks = Ref(0)
    _, errs1 = train_adjoint(model, generate_batch;
        learning_rate=3e-3, n_epochs=100, verbose=0, num_samples=64,
        patience=2000, lr_decay=0.5, min_lr=1e-6,
        on_epoch=(E, epoch, errors) -> (n_callbacks[] += 1),
        checkpoint_every=10, checkpoint_path=cp_path)
    @assert minimum(errs1) < errs1[1] * 0.9 "train_adjoint did not learn: $(errs1[1]) -> $(minimum(errs1))"
    @assert n_callbacks[] == 100
    @assert isfile(cp_path)

    model_b = init_model(t, 8, 4; sigma=tanh, RepType=Spline, init_scale=0.3)
    _, errs2 = train_adjoint(model_b, generate_batch;
        learning_rate=3e-3, n_epochs=10, verbose=0, num_samples=64,
        patience=2000, lr_decay=0.5, min_lr=1e-6,
        resume_from=cp_path)
    @assert length(errs2) > 10
    @assert minimum(errs2) <= errs2[1]
end

@slowtest "JLArray loss and gradient" begin
    JLArrays.allowscalar(false)
    Random.seed!(23)
    task = DeterminantTask(2)
    m = init_model(task, 4, 3; sigma=tanh, RepType=Spline, T=Float32)
    inputs, targets = generate(task, 4)
    Ain = Float32.(reduce(hcat, vec.(inputs)))
    Y = Float32.(reduce(hcat, targets))
    p = param_vector(m)
    E_cpu, g_cpu = batched_value_and_gradient(p, m, Ain, Y; dt=1.0f0 / 32)

    pj = adapt(JLArrays.JLArray, p)
    Aj = adapt(JLArrays.JLArray, Ain)
    Yj = adapt(JLArrays.JLArray, Y)
    E_j, g_j = batched_value_and_gradient(pj, m, Aj, Yj; dt=1.0f0 / 32)
    @assert isapprox(E_j, E_cpu; rtol=1e-4) "E JLArray vs CPU mismatch: $E_j vs $E_cpu"
    @assert isapprox(adapt(Array, g_j), g_cpu; rtol=1e-4, atol=1e-6) "gradient JLArray vs CPU mismatch: $(relerr(adapt(Array, g_j), g_cpu))"
    opt = AdamFlat(pj)
    apply_gradients!(opt, pj, 1e-3, g_j)
    @assert all(isfinite, adapt(Array, pj))
end

println("\nFiniteDiff")

@testcase "fd_loss" begin
    Random.seed!(21)
    model = small_model(3, 3)
    inputs = [randn(2, 2) for _ in 1:3]
    targets = [[det(M)] for M in inputs]
    E_fd = fd_loss(model, inputs, targets)
    E_ref = sum(_model_loss(model, inputs[n], targets[n], solve_tsit5) for n in eachindex(inputs))
    @assert isapprox(E_fd, E_ref; rtol=1e-9)
end

@slowtest "BatchedProblems gradient matches per-problem adjoint" begin
    Random.seed!(1)
    h = 5
    ncp = 3
    out = 1
    N = 8
    in_d = 6
    K = 3
    models = [NeuralFlowODE(
        Spline([0.3 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.3 .* randn(h) for _ in 1:ncp]),
        0.3 .* randn(h, in_d), 0.3 .* randn(out, h), tanh) for _ in 1:K]
    Ains = randn(in_d, N, K)
    Ys = [randn(out, N) for _ in 1:K]

    P = stack_group(models)
    Ystack = cat(Ys...; dims=3)
    E, ∇E = batched_group_value_and_gradient(P, Ains, Ystack,
        models[1].sigma, models[1].A, models[1].B, h, ncp, ncp;
        solver=Tsit5(), reltol=1e-10, abstol=1e-12)

    tot = 0.0
    for k in 1:K
        Ek, gk = NeuralFlow.Adjoint.batched_value_and_gradient(
            param_vector(models[k]), models[k], Ains[:, :, k], Ys[k];
            solver=Tsit5(), reltol=1e-10, abstol=1e-12)
        tot += Ek
        @assert relerr(∇E.A[:, :, :, k], gk.A) < 1e-6
        @assert relerr(∇E.B[:, :, k], gk.B) < 1e-6
        @assert relerr(∇E.W_in[:, :, k], gk.W_in) < 1e-6
        @assert relerr(∇E.W_out[:, :, k], gk.W_out) < 1e-6
    end
    @assert abs(E - tot) < 1e-6 * abs(tot)
end

@slowtest "train_batched groups by shape and reduces the objective" begin
    Random.seed!(7)
    tasks = [DeterminantTask(2), DeterminantTask(3)]
    models = [init_model(t, 12, 4; sigma=tanh, rng=Xoshiro(11)) for t in tasks]
    gens = [() -> generate(tasks[i], 32) for i in eachindex(tasks)]
    @assert length(group_by_shape(models, gens)) == 2
    _, errs = train_batched(models, gens; solver=Tsit5(), n_epochs=120,
        num_samples=32, learning_rate=1e-3, verbose=0, reltol=1e-4, abstol=1e-6)
    @assert all(length(e) == 120 for e in errs)
    @assert all(all(isfinite, e) for e in errs)
    @assert sum(e[end] for e in errs) < sum(e[1] for e in errs)
end

@slowtest "BatchedProblems fast RK4 adjoint matches finite differences" begin
    Random.seed!(2)
    h = 4
    ncp = 2
    out = 1
    N = 3
    nsteps = 8
    in_d = 5
    K = 2
    models = [NeuralFlowODE(
        Spline([0.3 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.3 .* randn(h) for _ in 1:ncp]),
        0.3 .* randn(h, in_d), 0.3 .* randn(out, h), tanh) for _ in 1:K]
    Ains = randn(in_d, N, K)
    Ystack = cat([randn(out, N) for _ in 1:K]...; dims=3)
    P = stack_group(models)
    σ = models[1].sigma
    repA = models[1].A
    repB = models[1].B
    _, ∇E = fast_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps)
    val(P2) = fast_group_value(P2, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps)
    @assert relerr(_flatP(∇E), _flatP(_fd_grad(val, P))) < 1e-5
end

@testcase "ResBlockFlow [1//1] reproduces NeuralFlowODE fast path" begin
    task = DeterminantTask(3)
    ins, tg = generate(task, 6; rng=Xoshiro(7))
    Ain = reduce(hcat, vec.(ins))
    Y = Float64.(reduce(hcat, [vec(t) for t in tg]))
    mb = init_resblock(task, 4, 2, [1 // 1]; sigma=relu, rng=Xoshiro(99))
    mo = init_model(task, 4, 2; sigma=relu, rng=Xoshiro(99))
    Eb, gb = block_fast_value_and_gradient(param_vector(mb), mb, Ain, Y; nsteps=16)
    Eo, go = fast_value_and_gradient(param_vector(mo), mo, Ain, Y; nsteps=16)
    @assert Eb == Eo
    @assert vec(gb.A) == vec(go.A) && vec(gb.B) == vec(go.B)
    @assert gb.W_in == go.W_in && gb.W_out == go.W_out
    # batched path vs the single-layer batched fast path
    rmods = [init_resblock(task, 4, 2, [1 // 1]; sigma=relu, rng=Xoshiro(300 + k)) for k in 1:2]
    omods = [init_model(task, 4, 2; sigma=relu, rng=Xoshiro(300 + k)) for k in 1:2]
    AinsK = cat(Ain, Ain; dims=3)
    YK = cat(Y, Y; dims=3)
    Pb = block_stack_group(rmods)
    Po = stack_group(omods)
    Eb2, gb2 = block_fast_group_value_and_gradient(Pb, AinsK, YK, rmods[1]; nsteps=16)
    Eo2, go2 = fast_group_value_and_gradient(Po, AinsK, YK, relu, omods[1].A, omods[1].B, 4, 2, 2; nsteps=16)
    @assert Eb2 == Eo2
    @assert vec(gb2.A1) == vec(go2.A) && vec(gb2.W_out) == vec(go2.W_out)
end

@testcase "ResBlockFlow depth-N adjoint matches FD" begin
    task = DeterminantTask(3)
    K = 2
    N = 3
    nsteps = 8
    gk = [generate(task, N; rng=Xoshiro(10 + k)) for k in 1:K]
    Ains = cat([reduce(hcat, vec.(gk[k][1])) for k in 1:K]...; dims=3)
    Ystack = cat([Float64.(reduce(hcat, [vec(t) for t in gk[k][2]])) for k in 1:K]...; dims=3)
    for mults in ([1 // 1], [1 // 1, 1 // 1], [1 // 2, 1 // 2, 1 // 1])
        models = [init_resblock(task, 4, 2, mults; sigma=tanh, rng=Xoshiro(100 + k)) for k in 1:K]
        P = block_stack_group(models)
        _, g = block_fast_group_value_and_gradient(P, Ains, Ystack, models[1]; nsteps)
        val(P2) = block_fast_group_value(P2, Ains, Ystack, models[1]; nsteps)
        @assert relerr(collect(g), collect(_fd_grad(val, P))) < 1e-5
        Esum = sum(block_fast_value_and_gradient(param_vector(models[k]), models[k],
            Ains[:, :, k], Ystack[:, :, k]; nsteps)[1] for k in 1:K)
        @assert relerr([block_fast_group_value(P, Ains, Ystack, models[1]; nsteps)], [Esum]) < 1e-10
    end
end

@slowtest "train_batched fast backends reduce the objective" begin
    backends = [(:fastrk4, (; nsteps=16)), (:fasteuler, (; nsteps=16)),
        (:fasttsit5, (; nsteps=8)), (:picard, (; nsteps=8)),
        (:fastadaptive, (; reltol=1e-3, abstol=1e-6))]
    for (backend, kw) in backends
        Random.seed!(7)
        tasks = [DeterminantTask(2), DeterminantTask(3)]
        models = [init_model(t, 12, 4; sigma=tanh, rng=Xoshiro(11)) for t in tasks]
        gens = [() -> generate(tasks[i], 32) for i in eachindex(tasks)]
        _, errs = train_batched(models, gens; n_epochs=120, num_samples=32,
            learning_rate=1e-3, verbose=0, backend=backend, kw...)
        @assert all(length(e) == 120 for e in errs) "$backend length"
        @assert all(all(isfinite, e) for e in errs) "$backend finite"
        @assert sum(e[end] for e in errs) < sum(e[1] for e in errs) "$backend reduces"
    end
end

@slowtest "BatchedProblems fast Euler adjoint matches finite differences" begin
    Random.seed!(2)
    h = 4
    ncp = 2
    out = 1
    N = 3
    nsteps = 8
    in_d = 5
    K = 2
    models = [NeuralFlowODE(
        Spline([0.3 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.3 .* randn(h) for _ in 1:ncp]),
        0.3 .* randn(h, in_d), 0.3 .* randn(out, h), tanh) for _ in 1:K]
    Ains = randn(in_d, N, K)
    Ystack = cat([randn(out, N) for _ in 1:K]...; dims=3)
    P = stack_group(models)
    σ = models[1].sigma
    repA = models[1].A
    repB = models[1].B
    _, ∇E = fast_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps, method=:euler)
    val(P2) = fast_group_value(P2, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps, method=:euler)
    @assert relerr(_flatP(∇E), _flatP(_fd_grad(val, P))) < 1e-5
end

@slowtest "BatchedProblems fasttsit5 adjoint matches finite differences" begin
    Random.seed!(2)
    h = 4
    ncp = 2
    out = 1
    N = 3
    nsteps = 6
    in_d = 5
    K = 2
    models = [NeuralFlowODE(
        Spline([0.3 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.3 .* randn(h) for _ in 1:ncp]),
        0.3 .* randn(h, in_d), 0.3 .* randn(out, h), tanh) for _ in 1:K]
    Ains = randn(in_d, N, K)
    Ystack = cat([randn(out, N) for _ in 1:K]...; dims=3)
    P = stack_group(models)
    σ = models[1].sigma
    repA = models[1].A
    repB = models[1].B
    _, ∇E = fast_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps, method=:tsit5)
    val(P2) = fast_group_value(P2, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps, method=:tsit5)
    @assert relerr(_flatP(∇E), _flatP(_fd_grad(val, P))) < 1e-5
end

@slowtest "BatchedProblems fastadaptive adjoint matches finite differences" begin
    BP = NeuralFlow.BatchedProblems
    Random.seed!(2)
    h = 4
    ncp = 2
    out = 1
    N = 3
    in_d = 5
    K = 2
    models = [NeuralFlowODE(
        Spline([0.3 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.3 .* randn(h) for _ in 1:ncp]),
        0.3 .* randn(h, in_d), 0.3 .* randn(out, h), tanh) for _ in 1:K]
    Ains = randn(in_d, N, K)
    Ystack = cat([randn(out, N) for _ in 1:K]...; dims=3)
    P = stack_group(models)
    σ = models[1].sigma
    repA = models[1].A
    repB = models[1].B
    rt, at = 1e-5, 1e-8
    _, ∇E = fastadaptive_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, h, ncp, ncp; reltol=rt, abstol=at)

    X0 = BP._embed_x0(P, Ains)
    _, _, hs0 = BP._t5_schedule(X0, P, σ, repA, repB, h, ncp, ncp; reltol=rt, abstol=at)
    function fixedval(P2)
        X = BP._embed_x0(P2, Ains)
        t = 0.0
        for hh in hs0
            X, _ = BP._t5_step(X, P2, t, hh, σ, repA, repB, h, ncp, ncp)
            t += hh
        end
        sum(abs2, BP.batched_mul(P2.W_out, X) .- Ystack)
    end
    @assert relerr(_flatP(∇E), _flatP(_fd_grad(fixedval, P))) < 1e-5
end

@slowtest "BatchedProblems Picard gradient matches finite differences" begin
    BP = NeuralFlow.BatchedProblems
    Random.seed!(3)
    h = 4
    ncp = 2
    out = 1
    N = 3
    in_d = 5
    K = 2
    nsteps = 4
    models = [NeuralFlowODE(
        Spline([0.3 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.3 .* randn(h) for _ in 1:ncp]),
        0.3 .* randn(h, in_d), 0.3 .* randn(out, h), tanh) for _ in 1:K]
    Ains = randn(in_d, N, K)
    Ystack = cat([randn(out, N) for _ in 1:K]...; dims=3)
    P = stack_group(models)
    σ = models[1].sigma
    repA = models[1].A
    repB = models[1].B
    _, ∇E = picard_group_value_and_gradient(P, Ains, Ystack, σ, repA, repB, h, ncp, ncp; nsteps)

    val(P2) = sum(abs2, BP.batched_mul(P2.W_out,
        BP._picard_final(P2, Ains, σ, repA, repB, h, ncp, ncp, nsteps)) .- Ystack)
    @assert relerr(_flatP(∇E), _flatP(_fd_grad(val, P))) < 1e-5
end

@slowtest "BatchedProblems Picard forward matches Tsit5" begin
    BP = NeuralFlow.BatchedProblems
    Random.seed!(5)
    h = 4
    ncp = 3
    in_d = 4
    N = 2
    K = 1
    model = NeuralFlowODE(
        Spline([0.2 .* randn(h, h) for _ in 1:ncp]),
        Spline([0.2 .* randn(h) for _ in 1:ncp]),
        0.2 .* randn(h, in_d), 0.2 .* randn(1, h), tanh)
    Ains = randn(in_d, N, K)
    P = stack_group([model])
    σ = model.sigma
    repA = model.A
    repB = model.B
    Xpic = BP._picard_final(P, Ains, σ, repA, repB, h, ncp, ncp, 64; piters=60)
    for n in 1:N
        a = model.W_in * Ains[:, n, 1]
        _, xs = solve_tsit5(model, a, [0.0, 1.0]; reltol=1e-9, abstol=1e-12)
        @assert relerr(Xpic[:, n, 1], xs[end]) < 1e-3
    end
end

@testcase "batched_heldout_errors: grouping-invariant, matches per-sample MSE" begin
    Random.seed!(3)
    tasks = [DeterminantTask(2), DeterminantTask(2), DeterminantTask(3)]
    models = [init_model(t, 10, 4; sigma=tanh, rng=Xoshiro(20 + i)) for (i, t) in enumerate(tasks)]
    N = 16
    gens = [() -> generate(tasks[i], N; rng=Xoshiro(100 + i)) for i in eachindex(tasks)]

    e = batched_heldout_errors(models, gens; backend=:fastrk4, nsteps=16)
    @assert length(e) == length(models) && all(isfinite, e) && all(>=(0), e)

    for i in eachindex(models)
        P = stack_group([models[i]])
        ins, tgts = (() -> generate(tasks[i], N; rng=Xoshiro(100 + i)))()
        Ains = reshape(Float64.(reduce(hcat, vec.(ins))), :, N, 1)
        Ystack = reshape(Float64.(reduce(hcat, vec.(tgts))), size(tgts[1], 1), N, 1)
        ref = fast_group_value(P, Ains, Ystack, models[i].sigma,
            models[i].A, models[i].B, 10, 4, 4; nsteps=16) / N
        @assert relerr([e[i]], [ref]) < 1e-10
    end

    # Grouping must not change results: det(2) models batch together, det(3) splits off.
    e1 = batched_heldout_errors(models[1:2], gens[1:2]; backend=:fastrk4, nsteps=16)
    @assert relerr(e1, e[1:2]) < 1e-12
end

println("\n" * "="^50)
println("Results: $passed passed, $failed failed, $skipped skipped")
println("="^50)
failed > 0 && exit(1)
