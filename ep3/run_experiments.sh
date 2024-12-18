#!/bin/bash

# Parâmetros de configuração
ITERATIONS=10
IMAGE_SIZES=(500 1000 5000)
PROCESS_COUNTS_MPICH=(2 4 8 10)
PROCESS_COUNTS_SIMGRID=(2 4 8 16 32 64)
SIMGRID_LATENCY=(10 100 500)
SEQ_EXEC="./sequential_julia"
MPICH_EXEC="./1D_parallel_julia"  # Executável MPICH
SIMGRID_EXEC="./1D_parallel_julia"
HETEROGENEOUS_CLUSTER="simple_cluster.xml"
HOSTFILE="simple_cluster_hostfile.txt"
RESULTS_DIR="results"

# Criação de diretórios
mkdir -p $RESULTS_DIR/sequential $RESULTS_DIR/mpich # $RESULTS_DIR/simgrid_homogeneous $RESULTS_DIR/simgrid_heterogeneous
make

run_seq() {
    echo "### Executando Testes Sequencial ###"
    for SIZE in "${IMAGE_SIZES[@]}"; do
        OUTPUT_FILE="$RESULTS_DIR/sequential/sequential_${SIZE}.txt"
        echo "Imagem: $SIZE"
            for ((i=1; i<=$ITERATIONS; i++)); do
                { time $SEQ_EXEC $SIZE ; } &>> "$OUTPUT_FILE"
            done
        
    done

}

# Função para executar testes com MPICH
run_mpich() {
    echo "### Executando Testes com MPICH ###"
    for SIZE in "${IMAGE_SIZES[@]}"; do
        for PROC in "${PROCESS_COUNTS_MPICH[@]}"; do
            OUTPUT_FILE="$RESULTS_DIR/mpich/mpich_${SIZE}_${PROC}.txt"
            echo "Imagem: $SIZE, Processos: $PROC"
            for ((i=1; i<=$ITERATIONS; i++)); do
                { time mpirun -np $PROC $MPICH_EXEC $SIZE ; } &>> "$OUTPUT_FILE"
            done
        done
    done
}


# Função para executar testes com MPICH
run_simgrid_heterogeneous() {
    echo "### Executando Testes com SIMGRID###"
    FIXED_SIZE=5000

    for PROC in "${PROCESS_COUNTS_SIMGRID[@]}"; do
        OUTPUT_FILE="$RESULTS_DIR/simgrid/heterogeneous_${PROC}.txt"
        echo "Imagem: $FIXED_SIZE, Processos: $PROC"
        for ((i=1; i<=$ITERATIONS; i++)); do
            { smpirun -np $PROC -hostfile $HOSTFILE -platform $HETEROGENEOUS_CLUSTER $SIMGRID_EXEC $FIXED_SIZE; } >> "$OUTPUT_FILE"
        done
    done
}

# Execução dos testes
# run_seq
# run_mpich
run_simgrid_heterogeneous

echo "### Todos os testes foram concluídos! ###"