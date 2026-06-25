#!/bin/sh
# Precompile the notebook env from the terminal. Run after src/ changes, before
# restarting the kernel
set -e
cd "$(dirname "$0")/.."
JULIA_LOAD_PATH="@:@stdlib" julia --project=notebooks -e 'using Pkg; Pkg.resolve(); Pkg.precompile()'
