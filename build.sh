#!/bin/bash
set -xe

CC="gcc"
CFLAGS="-Wall -Wextra -Wno-unused-result -O5"
INCLUDE_PATHS=("." "lodge")
LIBS=("m" "pthread")
DEMO="mnist"

if [ ! -d "mnist" ]; then
	mkdir mnist
    wget https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/train-images-idx3-ubyte.gz -P mnist
    gunzip mnist/train-images-idx3-ubyte.gz
	wget https://github.com/golbin/TensorFlow-MNIST/raw/refs/heads/master/mnist/data/train-labels-idx1-ubyte.gz -P mnist
    gunzip mnist/train-labels-idx1-ubyte.gz
fi

for dir in "${INCLUDE_PATHS[@]}"; do
	CFLAGS="${CFLAGS} -I${dir}"
done

LFLAGS=""
for lib in "${LIBS[@]}"; do
	LFLAGS="${LFLAGS} -l${lib}"
done

mkdir -p ./build
$CC "${DEMO}.c" $CFLAGS -o "./build/${DEMO}" $LFLAGS
./build/$DEMO
