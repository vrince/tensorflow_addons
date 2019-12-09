#!/bin/bash

# source .env/bin/activate
# pip install tensorflow-gpu==2.0.0-beta1
# configure.sh
# build.sh

set -e
rm ./artifacts/* || true
pip uninstall -y tensorflow_addons
bazel build build_pip_pkg
./build_pip_pkg.sh artifacts --nightly
pip install --force-reinstall artifacts/*.whl