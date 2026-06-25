"""
Handles saving and loading experiment results.
`Checkpointer` tracks the best model seen during training and periodically
snapshots it to disk so long runs can be resumed.
"""
module Results

using Serialization
using Dates: unix2datetime, format

export experiment_dir, save_run, load_run, Checkpointer, checkpoint!, on_improve!

const RESULTS_ROOT = joinpath(dirname(@__DIR__), "results")

"""
    experiment_dir(name) → String
"""
function experiment_dir(name::AbstractString)
    p = joinpath(RESULTS_ROOT, name)
    mkpath(p)
    return p
end
"""
    save_run(path; overwrite=false, kwargs...) → Nothing

Save all keyword arguments to `path`, e.g. save_run(path; model, errors, config).
"""
function save_run(path::AbstractString; overwrite::Bool=false, kwargs...)
    mkpath(dirname(path))
    overwrite || _archive_existing(path)
    data = NamedTuple(kwargs)
    # Write to a temp file, then rename into place, so an interrupt or
    # crash mid-write doesn't leave a partial file at `path`.
    tmp, io = mktemp(dirname(path))
    ok = false
    try
        serialize(io, data)
        ok = true
    finally
        close(io)
        ok || rm(tmp; force=true)
    end
    mv(tmp, path; force=true)
    return nothing
end

function _archive_existing(path::AbstractString)
    isfile(path) || return nothing
    ts = format(unix2datetime(mtime(path)), "yyyy-mm-ddTHHMMSS")
    base, ext = splitext(path)
    archive = "$base.$ts$ext"
    n = 1
    while isfile(archive)
        archive = "$base.$ts-$n$ext"
        n += 1
    end
    mv(path, archive)
    return nothing
end

"""
    load_run(path) → NamedTuple

Load a saved run as a NamedTuple.
"""
function load_run(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)::NamedTuple
    end
end


"""
    Checkpointer(path, every; metadata=NamedTuple())
"""
mutable struct Checkpointer
    path::String
    every::Int
    best::Float64
    snapshot_model::Any
    last_checkpoint_epoch::Int
    metadata::Dict{Symbol,Any}
    started::Bool
end

function Checkpointer(path::AbstractString, every::Int; metadata=NamedTuple())
    return Checkpointer(String(path), every, Inf, nothing, 0, Dict(pairs(metadata)), false)
end

"""
    on_improve!(c, E, epoch, model_snapshot)

Store the best model snapshot and its error.
"""
function on_improve!(c::Checkpointer, E::Real, epoch::Int, model_snapshot)
    c.best = Float64(E)
    c.snapshot_model = model_snapshot
    return nothing
end

"""
    checkpoint!(c, epoch, errors)

Write current state to disk, every `every` epochs.
"""
function checkpoint!(c::Checkpointer, epoch::Int, errors::AbstractVector)
    c.every > 0 || return nothing
    epoch - c.last_checkpoint_epoch >= c.every || return nothing
    c.last_checkpoint_epoch = epoch
    # archive a pre-existing file from an older run once, then update in place
    save_run(c.path; overwrite=c.started,
        best_model=c.snapshot_model, best_error=c.best,
        errors=collect(errors), epoch=epoch,
        metadata=c.metadata)
    c.started = true
    return nothing
end

end
