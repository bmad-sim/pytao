#!/bin/bash
set -e

# Important! Do not show full Bokeh server-backed applications in this
# converted output as they are not supported.
export PYTAO_BOKEH_NBCONVERT=1

REPO_ROOT=$(git rev-parse --show-toplevel)
PATTERN=${1-:"*"}
cd "$REPO_ROOT/docs/examples" || exit 1

NOTEBOOKS=$(git ls-files "${PATTERN}.ipynb")

SKIP_PATTERNS=()

# Silence Jupyterlab warning
export PYDEVD_DISABLE_FILE_VALIDATION=1

for file in $NOTEBOOKS; do
  should_skip=false
  for SKIP in "${SKIP_PATTERNS[@]}"; do
    if [[ "$file" == *"$SKIP"* ]]; then
      should_skip=true
      break
    fi
  done

  if [ "$should_skip" = true ]; then
    echo "* Skipping: $(basename "$file") as it matches a skip_pattern"
  else
    pushd "$(dirname "$file")" >/dev/null || exit
    echo "* Processing: $(basename "$file") in $PWD"
    jupyter nbconvert --to notebook --execute "$(basename "$file")" --inplace
    popd >/dev/null || exit
  fi
  echo
done
