#!/usr/bin/env bash

# Compile wheels
# sudo yum -y update
# curl -sL https://rpm.nodesource.com/setup_12.x | sudo bash -
curl -sL https://rpm.nodesource.com/setup_current.x | sudo bash -
# sudo yum clean all && sudo yum makecache fast
# sudo yum install -y gcc-c++ make
sudo yum install -y nodejs

rm -rf _wheelhouse
rm -rf _skbuild

# https://stackoverflow.com/a/40950971
for PYBIN in /opt/python/*/bin; do
    ver=$("$PYBIN"/python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
    # echo "$ver"
    if [ "$ver" -lt "38" ]; then
      continue
    fi
    # if [ "$ver" -gt "38" ]; then
    #   continue
    # fi

    sudo ${PYBIN}/pip install --upgrade -r ./dev-requirements.txt
    # ${PYBIN}/pip wheel . -w wheelhouse/
    ${PYBIN}/python setup.py bdist_wheel -d _wheelhouse
    rm -rf _skbuild
done

# Bundle external shared libraries into the wheels

for whl in _wheelhouse/*.whl; do
    auditwheel repair $whl -w ./dist/
done

rm -rf _wheelhouse
rm -rf _skbuild

# remove non bundled wheels
#for whl in dist/*.whl; do
#  if [[ "$whl" != *"manylinux"* ]];then
#    # printf '%s\n' "$whl"
#    rm $whl
#  fi
#done
