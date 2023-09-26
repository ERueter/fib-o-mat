#!/usr/bin/env bash

#rm -rf _skbuild
#rm -rf _wheelhouse

#declare -a CROSS_COMPILER_IMAGE_NAMES=("manylinux2014-x64")  # "manylinux2014-x86"
#
#for CROSS_COMPILER_IMAGE_NAME in "${CROSS_COMPILER_IMAGE_NAMES[@]}"; do
#  docker run --rm dockcross/${CROSS_COMPILER_IMAGE_NAME} > ./dockcross
#  chmod +x ./dockcross
#
#  ./dockcross bash scripts/build_linux_wheels_dockcross.sh
#  if [ "$CROSS_COMPILER_IMAGE_NAME" == "manylinux2014-x64" ]; then
#    ./dockcross bash scripts/build_linux_sdist.sh
#  fi
#done

docker run --rm dockcross/manylinux2014-x64 > ./dockcross
chmod +x ./dockcross
./dockcross bash scripts/build_linux_wheels_dockcross.sh
./dockcross bash scripts/build_linux_sdist.sh

rm dockcross
