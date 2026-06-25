# Shared CUDA helpers for the experiment scripts. Kept out of src/ so the package stays
# device-neutral; `include` this after `using CUDA, Adapt`.

is_gpu_oom(e) = e isa CUDA.OutOfGPUMemoryError ||
                (e isa TaskFailedException && is_gpu_oom(e.task.exception))

gpu_enabled(setting::AbstractString="auto") =
    setting != "0" && setting != "off" && CUDA.functional()

# A GPU run with >1 Julia thread can intermittently SIGSEGV
function warn_unsafe_gpu_threads()
    if Threads.nthreads() > 1
        @warn "GPU run with --threads=$(Threads.nthreads()): can intermittently SIGSEGV \
               Relaunch GPU runs with --threads=1." maxlog = 1
    end
end

function gpu_device(setting::AbstractString="auto")
    if gpu_enabled(setting)
        warn_unsafe_gpu_threads()
        return x -> adapt(CuArray, x)
    end
    return identity
end
