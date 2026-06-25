# Shared batched sweep utilities

"""
    BatchCtx

Per-sub-batch context passed to a sweep.
"""
struct BatchCtx
    batch::Vector{Int}
    models::Vector{Any}
    lr0::Union{Nothing,Vector{Float64}}
    lr_live::Union{Nothing,Vector{Float64}}
    save_cb::Union{Nothing,Function}
    record_lr::Function
    resume_lr::Float64
end

"""
    run_seed_sweep(seeds; ...) → (; models, lrs, final_lr, loss_curves, per_seed_curves)

Sweep `seeds` as batched solves with resume and periodic checkpointing.
"""
function run_seed_sweep(seeds; experiment, tag, seed_batch, per_seed_schedule::Bool,
    base_lr, checkpoint_every, lg, init_model, process,
    init_models=nothing, init_lr=nothing,
    make_payload=(weights, lr) -> (; weights, lr),
    started::Ref{Bool}=Ref(false), final_save::Bool=true, label::AbstractString=tag,
    oom_log=(msg -> logln(lg, msg)))

    base_lr = Float64(base_lr)
    final_lr = Ref(base_lr)

    warmw = init_models === nothing ? nothing :
            Dict(seeds[i] => init_models[i] for i in eachindex(seeds) if init_models[i] !== nothing)
    warm_lr = (init_lr isa AbstractVector) ?
              Dict(seeds[i] => Float64(init_lr[i]) for i in eachindex(seeds)) : nothing
    resume_lr = (init_lr isa Real) ? Float64(init_lr) : base_lr

    mdls = Dict{Int,Any}()
    lrd = Dict{Int,Float64}()

    prior_psc, prior_loss = let p = model_path(experiment, tag)
        if final_save && isfile(p)
            m = load_run(p).models
            psc = (hasproperty(m, :per_seed_curves) &&
                   length(m.per_seed_curves) == length(seeds)) ? m.per_seed_curves : nothing
            (psc, hasproperty(m, :loss_curves) ? collect(m.loss_curves) : NamedTuple[])
        else
            (nothing, NamedTuple[])
        end
    end
    perseed_d = Dict{Int,Vector{Float64}}()
    batch_curves = NamedTuple[]
    accumulated_perseed() = [vcat(prior_psc === nothing ? Float64[] : prior_psc[i],
                                  get(perseed_d, seeds[i], Float64[])) for i in eachindex(seeds)]
    accumulated_loss() = vcat(prior_loss, batch_curves)
    with_curves(base) = final_save ?
        merge(base, (; loss_curves=accumulated_loss(), per_seed_curves=accumulated_perseed())) : base

    weights_aligned(batch, models) = begin
        curw = Dict{Int,Any}(mdls)
        for (m, s) in enumerate(batch)
            curw[s] = models[m]
        end
        Any[get(curw, s, nothing) for s in seeds]
    end
    lr_aligned(batch, lr_live) =
        if per_seed_schedule
            curl = Dict{Int,Float64}(lrd)
            for (m, s) in enumerate(batch)
                curl[s] = lr_live[m]
            end
            Float64[get(curl, s, base_lr) for s in seeds]
        else
            final_lr[]
        end

    function run_batch(batch)
        models = Any[(warmw !== nothing && haskey(warmw, s)) ? warmw[s] : init_model(s) for s in batch]
        lr_live = per_seed_schedule ?
            Float64[(warm_lr !== nothing && haskey(warm_lr, s)) ? warm_lr[s] : base_lr for s in batch] :
            nothing
        lr0 = lr_live === nothing ? nothing : copy(lr_live)
        record_lr = (E, ep, lr) -> (final_lr[] = lr; nothing)

        save_cb = periodic_save_models(experiment, tag, checkpoint_every; started,
            payload=() -> with_curves(make_payload(weights_aligned(batch, models), lr_aligned(batch, lr_live))))

        process(BatchCtx(batch, models, lr0, lr_live, save_cb, record_lr, resume_lr))

        for (m, s) in enumerate(batch)
            mdls[s] = models[m]
            per_seed_schedule && (lrd[s] = lr_live[m])
        end
    end

    sink = CurveSink()
    nseen = Ref(0)
    harvest!(batch) = begin
        all = take_curves(sink)
        for c in @view all[(nseen[]+1):end]
            if c.kind === :per_seed && c.tag isa Integer && 1 <= c.tag <= length(batch)
                perseed_d[batch[c.tag]] = c.curve
            elseif c.kind === :loss
                push!(batch_curves, (; seeds=copy(batch), K=length(batch), curve=c.curve))
            end
        end
        nseen[] = length(all)
    end

    todo = reverse([collect(c) for c in Iterators.partition(collect(seeds), seed_batch)])
    with_curve_sink(sink) do
        while !isempty(todo)
            batch = pop!(todo)
            try
                run_batch(batch)
                harvest!(batch)
            catch e
                (CUDA.functional() && is_gpu_oom(e) && length(batch) > 1) || rethrow()
                GC.gc()
                CUDA.reclaim()
                mid = cld(length(batch), 2)
                oom_log("  [$label] OOM on $(length(batch)) seeds, retry $mid + $(length(batch) - mid)")
                push!(todo, batch[mid+1:end])
                push!(todo, batch[1:mid])
            end
        end
    end

    per_seed_curves = accumulated_perseed()

    if final_save
        wfin = Any[get(mdls, s, nothing) for s in seeds]
        lfin = per_seed_schedule ? Float64[get(lrd, s, base_lr) for s in seeds] : final_lr[]
        save_run(model_path(experiment, tag); overwrite=started[],
            models=with_curves(make_payload(wfin, lfin)))
    end
    return (; models=mdls, lrs=lrd, final_lr=final_lr[],
        loss_curves=accumulated_loss(), per_seed_curves)
end

"""
    _split_calc_errors(task, hidden, ncp, seeds, split; ...) → (errors, samples, models, lr, epoch)

Train one split across all seeds.
"""
function _split_calc_errors(task, hidden, ncp, seeds, split; data_to_device, lg, label,
    per_seed_schedule::Bool, cfg, learning_rate, init_models, init_lrs, init_samples, ckpt, cleanup,
    n_epochs=cfg.nepochs)

    errs = Dict{Int,Float64}()

    samps = Dict{Int,Vector{Float64}}()
    live_curve = Dict{Int,Vector{Float64}}()
    prior_s = init_samples === nothing ? nothing :
              Dict(seeds[i] => init_samples[i] for i in eachindex(seeds))
    _prior(s) = prior_s === nothing ? Float64[] : get(prior_s, s, Float64[])
    _session(s) = haskey(samps, s) ? samps[s] : get(live_curve, s, Float64[])

    cumsamples() = [vcat(_prior(s), _session(s)) for s in seeds]
    cumepoch(cs) = cfg.sample_every * maximum(length, cs; init=0)
    ch, logtask = loss_logger(lg)
    nsteps = euler_steps(ncp)

    make_payload = (weights, lr) -> begin
        cs = cumsamples()
        (; weights=merge(ckpt.prior_weights, Dict(ckpt.split => weights)),
           lr=merge(ckpt.prior_lrs, Dict(ckpt.split => lr)),
           samples=merge(ckpt.prior_samples, Dict(ckpt.split => cs)),
           epoch=merge(ckpt.prior_epochs, Dict(ckpt.split => cumepoch(cs))),
           errors=ckpt.prior_errors)
    end

    process = ctx -> begin
        batch = ctx.batch
        models = ctx.models
        Kb = length(batch)
        gens = [
            let r = Xoshiro(s + DATA_OFFSET)
                () -> generate(task, cfg.nsamp; rng=r)
            end for s in batch
        ]
        on_loss = (E, ep, lr) -> begin
            ctx.record_lr(E, ep, lr)
            ep % cfg.log_every == 0 &&
                put!(ch, @sprintf("    %s epoch %6d  E=%.6e  lr=%.2e",
                    label, ep, E / (Kb * cfg.nsamp), lr))
        end
        on_eval = (it, perseed) -> for (m, s) in enumerate(batch)
            push!(get!(live_curve, s, Float64[]), perseed[m] / cfg.nsamp)
        end
        plateau = (!per_seed_schedule && cfg.plateau_evals > 0) ?
            let best = Ref(Inf), stale = Ref(0)
                (E, ep, _e) -> begin
                    if E < best[] * (1 - cfg.min_improve)
                        best[] = E
                        stale[] = 0
                    else
                        stale[] += 1
                    end
                    stale[] >= cfg.plateau_evals || return nothing
                    put!(ch, @sprintf("    %s plateau stop at epoch %6d  (best E=%.6e)",
                        label, ep, best[] / (Kb * cfg.nsamp)))
                    :stop
                end
            end : nothing
        on_epoch = combine_callbacks(plateau, ctx.save_cb)

        _, errs_hist, _ = train_batched(models, gens;
            n_epochs=n_epochs, num_samples=cfg.nsamp,
            learning_rate=ctx.resume_lr, patience=cfg.patience, lr_decay=cfg.lr_decay, min_lr=cfg.min_lr,
            data_to_device, reltol=cfg.reltol, abstol=cfg.abstol, dt=nothing,
            eval_every=cfg.sample_every, sync_every=cfg.sample_every, verbose=0,
            backend=split.backend, nsteps, collect_sample_errors=false, on_loss, on_epoch, on_eval,
            per_seed_schedule=per_seed_schedule, train_win=get(cfg, :train_win, false),
            plateau_evals=(per_seed_schedule ? cfg.plateau_evals : 0),
            min_improve=cfg.min_improve, lr0=ctx.lr0, lr_out=ctx.lr_live)

        cleanup()

        eval_gens = [
            let r = Xoshiro(s + EVAL_OFFSET)
                () -> generate(task, cfg.eval_n; rng=r)
            end for s in batch
        ]
        ek = batched_heldout_errors(models, eval_gens; backend=split.backend, nsteps,
            reltol=cfg.reltol, abstol=cfg.abstol, data_to_device)
        for (m, s) in enumerate(batch)
            errs[s] = ek[m]
            samps[s] = errs_hist[m] ./ cfg.nsamp
        end
        cleanup()
    end

    local res
    try
        res = run_seed_sweep(seeds; experiment=ckpt.experiment, tag=ckpt.tag,
            seed_batch=cfg.seed_batch, per_seed_schedule, base_lr=learning_rate,
            checkpoint_every=ckpt.every, lg, label,
            init_model=(s -> init_model(task, hidden, ncp; sigma=tanh, RepType=split.rep,
                init_scale=0.3, T=cfg.T, rng=Xoshiro(s))),
            process, init_models, init_lr=init_lrs, make_payload,
            started=ckpt.started, final_save=false, oom_log=(msg -> put!(ch, msg)))
    finally
        close(ch)
        wait(logtask)
    end

    cs = cumsamples()
    lr_out = per_seed_schedule ? Float64[get(res.lrs, s, Float64(learning_rate)) for s in seeds] : res.final_lr
    return [errs[s] for s in seeds], cs, [res.models[s] for s in seeds], lr_out, cumepoch(cs)
end

"""
    run_split_sweep(task, splits; ...) → (; names, errors, samples, trained, lrs, epochs)

Train every split across all seeds, with resume and periodic checkpointing.
"""
function run_split_sweep(task, splits; seeds, hidden, ncp, experiment, tag,
    data_to_device, lg, per_seed_schedule::Bool, cfg, cleanup=() -> nothing)

    base_lr = Float64(cfg.lr)
    names = [sp.name for sp in splits]

    warm = load_models(experiment, tag)
    warm !== nothing &&
        logln(lg, "  warm-starting from saved weights: $(join(keys(warm.weights), ", "))")
    warm_samples(name) = (warm === nothing || !hasproperty(warm, :samples)) ?
                         nothing : get(warm.samples, name, nothing)
    warm_epoch(name) = (warm === nothing || !hasproperty(warm, :epoch)) ?
                       0 : Int(get(warm.epoch, name, 0))
    warm_errors(name) = (warm === nothing || !hasproperty(warm, :errors)) ?
                        nothing : get(warm.errors, name, nothing)

    errors = Vector{Vector{Float64}}(undef, length(splits))
    samples = Vector{Any}(undef, length(splits))
    trained = Dict{String,Any}()
    lrs = Dict{String,Any}()
    samples_acc = Dict{String,Any}()
    epochs = Dict{String,Any}()
    errs_acc = Dict{String,Any}()
    trained_any = false
    started = Ref(false)

    for (j, sp) in enumerate(splits)
        wm = warm === nothing ? nothing : get(warm.weights, sp.name, nothing)
        wlr = warm === nothing ? nothing : get(warm.lr, sp.name, nothing)
        prior_ep = warm_epoch(sp.name)
        we = warm_errors(sp.name)

        if wm !== nothing && we !== nothing && prior_ep >= cfg.nepochs
            logln(lg, "  $(sp.name): complete ($prior_ep ≥ $(cfg.nepochs) epochs), skipping")
            ws = warm_samples(sp.name)
            errors[j] = collect(Float64, we)
            samples[j] = ws === nothing ? [Float64[] for _ in seeds] : ws
            trained[sp.name] = wm
            lrs[sp.name] = wlr === nothing ? base_lr : wlr
            samples_acc[sp.name] = samples[j]
            epochs[sp.name] = prior_ep
            errs_acc[sp.name] = errors[j]
            continue
        end
        lr_eff = (!per_seed_schedule && wlr isa Real) ? Float64(wlr) : base_lr
        init_lrs = (per_seed_schedule && wlr isa AbstractVector) ? wlr : nothing
        ckpt = (; experiment, tag, split=sp.name, prior_weights=trained, prior_lrs=lrs,
            prior_samples=samples_acc, prior_epochs=epochs, prior_errors=errs_acc,
            started, every=cfg.checkpoint_every)
        remaining = max(0, cfg.nepochs - prior_ep)
        wm !== nothing && remaining < cfg.nepochs &&
            logln(lg, "  $(sp.name): $prior_ep epochs trained, $remaining remaining to reach $(cfg.nepochs)")
        errors[j], samples[j], trained[sp.name], lrs[sp.name], epochs[sp.name] = _split_calc_errors(
            task, hidden, ncp, seeds, sp; data_to_device, lg, label="$tag $(sp.name)",
            per_seed_schedule, cfg, learning_rate=lr_eff, init_models=wm, init_lrs,
            init_samples=warm_samples(sp.name), ckpt, cleanup, n_epochs=remaining)
        samples_acc[sp.name] = samples[j]
        errs_acc[sp.name] = errors[j]
        trained_any = true
    end

    if trained_any || warm === nothing
        save_run(model_path(experiment, tag); overwrite=started[],
            models=(; weights=trained, lr=lrs, samples=samples_acc, epoch=epochs, errors=errs_acc))
    end

    return (; names, errors, samples, trained, lrs, epochs)
end
