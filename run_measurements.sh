#! /bin/bash

set -o xtrace
mkdir results

MEASUREMENTS=10
ITERATIONS=8
INITIAL_SIZE=16

SIZE=$INITIAL_SIZE

NAMES=('mandelbrot_seq' 'mandelbrot_pth' 'mandelbrot_omp')

make
mkdir results/input_size

for NAME in ${NAMES[@]}; do
    mkdir results/input_size/$NAME

    for ((i=1; i<=$ITERATIONS; i++)); do
            perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE >> full.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME -0.8 -0.7 0.05 0.15 $SIZE >> seahorse.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME 0.175 0.375 -0.1 0.1 $SIZE >> elephant.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME -0.188 -0.012 0.554 0.754 $SIZE >> triple_spiral.log 2>&1
            SIZE=$(($SIZE * 2))
    done

    SIZE=$INITIAL_SIZE

    mv *.log results/input_size/$NAME
    rm output.ppm
done

ITERATIONS=6
INITIAL_THREADS=1

SIZE=512
THREADS=$INITIAL_THREADS

NAMES=('mandelbrot_pth' 'mandelbrot_omp')
mkdir results/threads

for NAME in ${NAMES[@]}; do
    mkdir results/threads/$NAME

    for ((i=1; i<=$ITERATIONS; i++)); do
            perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE $THREADS >> full.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME -0.8 -0.7 0.05 0.15 $SIZE $THREADS >> seahorse.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME 0.175 0.375 -0.1 0.1 $SIZE $THREADS >> elephant.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME -0.188 -0.012 0.554 0.754 $SIZE $THREADS >> triple_spiral.log 2>&1
            THREADS=$(($THREADS * 2))
    done

    THREADS=$INITIAL_THREADS

    mv *.log results/threads/$NAME
    rm output.ppm
done

make clean