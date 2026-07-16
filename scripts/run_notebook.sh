#!/usr/bin/env bash
# Runs experiments/*.ipynb notebooks headlessly, in place, via the
# `audio-lab` conda env's Jupyter kernel (see experiments/README.md ->
# Setup). Each notebook downloads/repackages its own model artifacts
# into models/<stage>/ and data/eval/ as a side effect -- this is how
# those get (re)generated, see models/README.md.
#
# Usage:
#   scripts/run_notebook.sh                    # all notebooks, in order
#   scripts/run_notebook.sh 02_asr              # just one (.ipynb optional)
#   scripts/run_notebook.sh 02_asr.ipynb 03_mt  # a few
#
# Env vars:
#   ONEVOICE_NB_TIMEOUT   per-notebook execution timeout in seconds
#                         (default 1200 -- these download models and run
#                         real inference, plain "it's slow" is normal)
#   ONEVOICE_NB_KERNEL    Jupyter kernel name to execute with (default
#                         "audio-lab")

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$REPO_ROOT/experiments"
KERNEL="${ONEVOICE_NB_KERNEL:-audio-lab}"
TIMEOUT="${ONEVOICE_NB_TIMEOUT:-1200}"

if ! command -v jupyter >/dev/null 2>&1; then
    echo "error: 'jupyter' not found on PATH. See experiments/README.md -> Setup." >&2
    exit 1
fi

if ! jupyter kernelspec list 2>/dev/null | grep -qw "$KERNEL"; then
    echo "error: Jupyter kernel '$KERNEL' isn't registered." >&2
    echo "  conda activate $KERNEL" >&2
    echo "  pip install -r experiments/requirements.txt" >&2
    echo "  python -m ipykernel install --user --name $KERNEL --display-name \"Python ($KERNEL)\"" >&2
    exit 1
fi

cd "$EXPERIMENTS_DIR"

notebooks=()
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        case "$arg" in
            *.ipynb) notebooks+=("$arg") ;;
            *) notebooks+=("${arg}.ipynb") ;;
        esac
    done
else
    while IFS= read -r nb; do
        notebooks+=("$nb")
    done < <(find . -maxdepth 1 -name '*.ipynb' -print | sed 's|^\./||' | sort)
fi

if [ "${#notebooks[@]}" -eq 0 ]; then
    echo "no notebooks found in $EXPERIMENTS_DIR" >&2
    exit 1
fi

echo "kernel: $KERNEL  timeout: ${TIMEOUT}s"
echo "notebooks: ${notebooks[*]}"
echo

failed=()
for nb in "${notebooks[@]}"; do
    if [ ! -f "$nb" ]; then
        echo "!! $nb: not found, skipping"
        failed+=("$nb (not found)")
        continue
    fi

    echo "=== $nb ==="
    if ! jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout="$TIMEOUT" \
            --ExecutePreprocessor.kernel_name="$KERNEL" \
            "$nb"; then
        echo "=== FAILED (nbconvert error): $nb ==="
        failed+=("$nb")
        continue
    fi

    # nbconvert exits 0 even when a cell inside the notebook raised an
    # error (it just records the traceback in that cell's output) -- if
    # we don't check for that explicitly, a silently-broken notebook
    # looks like a pass.
    if python3 -c "
import sys
import nbformat
nb = nbformat.read('$nb', as_version=4)
had_error = any(
    out.get('output_type') == 'error'
    for cell in nb.cells if cell.cell_type == 'code'
    for out in cell.get('outputs', [])
)
sys.exit(1 if had_error else 0)
"; then
        echo "=== OK: $nb ==="
    else
        echo "=== FAILED (cell error inside notebook): $nb ==="
        failed+=("$nb")
    fi
    echo
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi

echo "all notebooks ran clean"
