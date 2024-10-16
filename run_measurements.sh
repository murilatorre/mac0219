#! /bin/bash

set -o xtrace

make
mkdir results
MEASUREMENTS=10

# Experimentos variando o tamanho da entrada
ITERATIONS=8
INITIAL_SIZE=16

SIZE=$INITIAL_SIZE

NAMES=('mandelbrot_seq' 'mandelbrot_pth' 'mandelbrot_omp')

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
done


# Experimentos sem/com alocação de memória para mandelbrot_seq
ITERATIONS=6

STATES=('with' 'without')
VALUE=0

mkdir results/io_aloc

for STATE in "${STATES[@]}"; do
    mkdir results/io_aloc/$STATE

    if [ $STATE == "with" ]; then
        VALUE=1
    else
        VALUE=0
    fi

    for ((i=1; i<=ITERATIONS; i++)); do
        perf stat -r $MEASUREMENTS ./mandelbrot_seq -2.5 1.5 -2.0 2.0 $SIZE $VALUE >> full.log 2>&1
        perf stat -r $MEASUREMENTS ./mandelbrot_seq -0.8 -0.7 0.05 0.15 $SIZE $VALUE >> seahorse.log 2>&1
        perf stat -r $MEASUREMENTS ./mandelbrot_seq 0.175 0.375 -0.1 0.1 $SIZE $VALUE >> elephant.log 2>&1
        perf stat -r $MEASUREMENTS ./mandelbrot_seq -0.188 -0.012 0.554 0.754 $SIZE $VALUE >> triple_spiral.log 2>&1
        SIZE=$((SIZE * 2))
    done

    SIZE=$INITIAL_SIZE
    mv *.log results/io_aloc/$STATE
done


# Experimentos variando o número de Threads
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
done

make clean
