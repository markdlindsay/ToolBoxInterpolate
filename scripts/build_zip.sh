#!/bin/sh
# Build a clean, installable plugin zip (no caches, no tests) in dist/.
set -e
cd "$(dirname "$0")/.."
version=$(sed -n 's/^version=//p' interpolation_toolbox/metadata.txt)
mkdir -p dist
out="dist/interpolation_toolbox-$version.zip"
rm -f "$out"
zip -rq "$out" interpolation_toolbox -x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
echo "Built $out"
