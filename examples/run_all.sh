#!/usr/bin/env bash

# https://stackoverflow.com/a/4774063
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

for file in "$SCRIPTPATH"/*.py ; do
  echo "Running $file"
  python "$file"
done
