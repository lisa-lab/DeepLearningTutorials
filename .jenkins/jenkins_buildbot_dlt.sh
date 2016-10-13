#!/bin/bash

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

BUILDBOT_DIR=$WORKSPACE/nightly_build

mkdir -p ${BUILDBOT_DIR}

date
COMPILEDIR=$WORKSPACE/compile/lisa_theano_compile_dir_deeplearning
NOSETESTS=${BUILDBOT_DIR}/Theano/bin/theano-nose
XUNIT="--with-xunit --xunit-file="
# name test suites
SUITE="--xunit-testsuite-name="

FLAGS=warn.ignore_bug_before=0.5,compiledir=${COMPILEDIR}
export PYTHONPATH=${BUILDBOT_DIR}/Theano:${BUILDBOT_DIR}/Pylearn:$PYTHONPATH

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

echo "executing nosetests speed with mode=FAST_RUN"
NAME=dlt_speed
FILE=${BUILDBOT_DIR}/${NAME}_tests.xml
THEANO_FLAGS=${FLAGS},mode=FAST_RUN ${NOSETESTS} ${XUNIT}${FILE} ${SUITE}${NAME} test.py:speed
echo "executing nosetests with mode=FAST_RUN,floatX=float32"
NAME=dlt_float32
FILE=${BUILDBOT_DIR}/${NAME}_tests.xml
THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32 ${NOSETESTS} ${XUNIT}${FILE} ${SUITE}${NAME}
