#!/bin/bash
set -e
rm ./artifacts/* || true
pip uninstall -y tensorflow_addons
bazel build build_pip_pkg
./build_pip_pkg.sh artifacts
pip install --force-reinstall artifacts/*.whl