#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

usage() {
  echo "Usage: $0 [BMAD_CONDA_VERSION]"
  echo "  BMAD_CONDA_VERSION  Minimum bmad conda version (default: 'show version' date, e.g. 20260710.0)"
}

case "$1" in
-h | --help)
  usage
  exit 0
  ;;
esac

if [ $# -gt 1 ]; then
  usage
  exit 1
fi

bmad_conda_version=$1

cd "$SCRIPT_DIR" || exit 1

set -e

echo "* Dumping structure information from Bmad..."
bash dump_structs_json.sh

echo "* Regenerating structures..."
python -m pytao.model.codegen structs.json

set +e

if command -v gsed 2>&1 >/dev/null; then
  # Prefer GNU sed on OSX
  SED=$(command -v gsed)
else
  SED=sed
fi

datetime=$(python "$SCRIPT_DIR"/tao_datetime_version.py)
if [ -z "$datetime" ]; then
  echo "Unable to determine version using pytao"
  exit 1
fi

# Conda package versions are the bmad git tag with '-' replaced by '.'
# (e.g., 20260710-0 -> conda 20260710.0)
if [ -z "$bmad_conda_version" ]; then
  # shellcheck disable=SC2046
  bmad_conda_version=$(printf "%04d%02d%02d.0" $(echo "$datetime" | "$SED" -E 's/[^0-9]+/ /g'))
fi

echo "Updating minimum version to:"
echo "* Conda: $bmad_conda_version"
echo "* Bmad 'show version': $datetime"

cd $SCRIPT_DIR/.. || exit 1

"$SED" -i -e "s/^\(\s*_min_tao_version =\).*/\1 ${datetime}/" pytao/interface.tpl.py
"$SED" -i -e "s/^\(\s*- bmad\) \s*.*/\1 >=${bmad_conda_version}/" environment.yml

cd $SCRIPT_DIR || exit 1
python generate_interface_commands.py

BMAD_TAG=$(echo "$bmad_conda_version" | $SED -e 's/\./-/')
echo "git commit -am 'MAINT: regenerate for Bmad $BMAD_TAG'"
