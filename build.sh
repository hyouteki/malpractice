#!/bin/bash

set -xe

CC="gcc"
DEMO="mnist"
LIBS="-lm"

mkdir -p ./build
$CC $DEMO.c -o ./build/$DEMO $LIBS
./build/$DEMO
