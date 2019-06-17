bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
echo y > /tmp/y
pip uninstall tensorflow < /tmp/y
pip install /tmp/tensorflow_pkg/tensorflow-1.11.0-cp27-cp27mu-linux_x86_64.whl
