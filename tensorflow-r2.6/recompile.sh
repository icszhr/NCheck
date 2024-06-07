pip uninstall tensorflow -y &&
bazel build --jobs=32 -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package &&
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg &&
pip install /tmp/tensorflow_pkg/tensorflow-2.6.5-cp39-cp39-linux_x86_64.whl
#kill -9 `pgrep python`
