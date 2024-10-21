#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

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
"$SED" -i -e "s/^\(\s*- bmad\) \s*.*/\1 >=${bmad_conda_version}/" environment.yml dev-environment.yml
python generate_interface_commands.py
