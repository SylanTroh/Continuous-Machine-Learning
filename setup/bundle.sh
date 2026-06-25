#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${1:-project_bundle.zip}"

rm -f -- "$OUT"
zip -r "$OUT" \
    src \
    Figures \
    Latex/NeuralFlowSlides.tex \
    Latex/NeuralFlowSlides.pdf \

echo
echo "Wrote $ROOT/$OUT"
