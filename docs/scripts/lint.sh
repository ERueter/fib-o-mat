#!/usr/bin/env sh

if [[ $# -ne 1 ]] ; then
    echo 'Expecting path as argument'
    exit 1
fi

file=$1

echo "mypy"
mypy $file

echo ""
echo "flake8"
flake8 $file

echo ""
echo "pylint"
pylint $file

echo ""
echo "darglint"
darglint $file

echo ""
echo "pydocstyle"
pydocstyle $file

