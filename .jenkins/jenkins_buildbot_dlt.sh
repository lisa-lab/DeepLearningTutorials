#!/bin/bash

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

# MKL
export MKL_THREADING_LAYER=GNU

# Set OpenMP threads for stability of speedtests
export OMP_NUM_THREADS=1

BUILDBOT_DIR=$WORKSPACE/nightly_build

mkdir -p ${BUILDBOT_DIR}

date
COMPILEDIR=$HOME/.theano/lisa_theano_buildbot_deeplearning
NOSETESTS=${BUILDBOT_DIR}/Theano/bin/theano-nose
XUNIT="--with-xunit --xunit-file="
# name test suites
SUITE="--xunit-testsuite-name="

FLAGS=warn.ignore_bug_before=0.5,compiledir=${COMPILEDIR}
export PYTHONPATH=${BUILDBOT_DIR}/Theano:${BUILDBOT_DIR}/Pylearn:$PYTHONPATH

# Install libgpuarray and pygpu
cd ${BUILDBOT_DIR}

# Make fresh clone (with no history since we don't need it)
rm -rf libgpuarray
git clone "https://github.com/Theano/libgpuarray.git"

(cd libgpuarray && echo "libgpuarray commit" && git rev-parse HEAD)

# Clean up previous installs (to make sure no old files are left)
rm -rf local
mkdir local

# Build libgpuarray and run C tests
mkdir libgpuarray/build
(cd libgpuarray/build && cmake .. -DCMAKE_BUILD_TYPE=${GPUARRAY_CONFIG} -DCMAKE_INSTALL_PREFIX=${BUILDBOT_DIR}/local && make)

# Finally install
(cd libgpuarray/build && make install)
export LD_LIBRARY_PATH=${BUILDBOT_DIR}/local/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${BUILDBOT_DIR}/local/lib:${LIBRARY_PATH}
export CPATH=${BUILDBOT_DIR}/local/include:${CPATH}

# Build the pygpu modules
(cd libgpuarray && python setup.py build_ext --inplace -I${BUILDBOT_DIR}/local/include -L${BUILDBOT_DIR}/local/lib)

mkdir ${BUILDBOT_DIR}/local/lib/python
export PYTHONPATH=${PYTHONPATH}:${BUILDBOT_DIR}/local/lib/python
# Then install
(cd libgpuarray && python setup.py install --home=${BUILDBOT_DIR}/local)

# Install Theano
cd ${BUILDBOT_DIR}
if [ ! -d ${BUILDBOT_DIR}/Theano ]; then
  git clone git://github.com/Theano/Theano.git
fi
# update repo
cd ${BUILDBOT_DIR}/Theano; git pull

cd ${WORKSPACE}/data
./download.sh

cd ${BUILDBOT_DIR}/Theano
echo "git version for Theano:" `git rev-parse HEAD`
cd ${WORKSPACE}/code
echo "git version:" `git rev-parse HEAD`

echo "==== Executing nosetests speed with mode=FAST_RUN"
NAME=dlt_speed
FILE=${BUILDBOT_DIR}/${NAME}_tests.xml
THEANO_FLAGS=${FLAGS},mode=FAST_RUN ${NOSETESTS} ${XUNIT}${FILE} ${SUITE}${NAME} test.py:speed

echo "==== Executing nosetests with mode=FAST_RUN,floatX=float32"
NAME=dlt_float32
FILE=${BUILDBOT_DIR}/${NAME}_tests.xml
THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32 ${NOSETESTS} ${XUNIT}${FILE} ${SUITE}${NAME}

echo "==== Executing nosetests with mode=FAST_RUN,floatX=float32,device=cuda"
NAME=dlt_float32_cuda
FILE=${BUILDBOT_DIR}/${NAME}_tests.xml
PYTHONPATH=${BUILDBOT_DIR}/Theano:${BUILDBOT_DIR}/DeepLearningTutorials/code:${PYTHONPATH} THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32,device=cuda nosetests test.py ${XUNIT}${FILE} ${SUITE}${NAME}
