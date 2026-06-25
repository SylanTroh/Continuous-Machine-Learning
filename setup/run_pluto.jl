#!/usr/bin/env julia
# Launch the Pluto demo (notebooks/Demo.jl).

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "pluto")))

import Pluto
Pluto.run(notebook=normpath(joinpath(@__DIR__, "..", "notebooks", "Demo.jl")))
