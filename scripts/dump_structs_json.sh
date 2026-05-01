#!/bin/bash

set -xe

if [ ! -d cppbmad ]; then
  git clone --depth 1 https://github.com/ken-lauer/cppbmad
else
  (cd cppbmad && git pull)
fi

if [ ! -d bmad ]; then
  git clone --depth 1 https://github.com/bmad-sim/bmad-ecosystem bmad
else
  (cd bmad && git pull)
fi

ACC_ROOT_DIR=$PWD/bmad
(cd cppbmad && python -m codegen.structs >../structs.json)
