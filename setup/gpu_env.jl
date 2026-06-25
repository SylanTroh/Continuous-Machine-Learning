# Per-machine GPU settings from local.toml
# Include BEFORE `using CUDA`; the memory limit is read once at pool init.
using TOML

let path = joinpath(@__DIR__, "..", "local.toml")
    if isfile(path)
        config = TOML.parsefile(path)
        if haskey(config, "vram_limit")
            ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = config["vram_limit"]
        end
    end
end
