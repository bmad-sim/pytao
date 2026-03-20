#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

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
bmad_conda_version=$(conda list --json | jq -r '.[] | select(.name=="bmad") | .version' | $SED -e 's/\..*$//')

if [ -z "$datetime" ]; then
  echo "Unable to determine version using pytao"
  exit 1
fi
if [ -z "$bmad_conda_version" ]; then
  echo "Unable to determine version using 'conda'"
  exit 1
fi

echo "Updating minimum version to:"
echo "* Conda: $bmad_conda_version"
echo "* Bmad 'show version': $datetime"

cd $SCRIPT_DIR/.. || exit 1

"$SED" -i -e "s/^\(\s*_min_tao_version =\).*/\1 ${datetime}/" pytao/interface.tpl.py
"$SED" -i -e "s/^\(\s*- bmad\) \s*.*/\1 >=${bmad_conda_version}/" environment.yml

cd $SCRIPT_DIR || exit 1
python generate_interface_commands.py

if [ -n "$bmad_conda_version" ]; then
  BMAD_TAG="$bmad_conda_version"
else
  BMAD_TAG=$(cd "$ACC_ROOT_DIR" && git describe --tags)
fi

if [ -n "$BMAD_TAG" ]; then
  echo "git commit -am 'MAINT: regenerate for Bmad $BMAD_TAG'"
fi
