import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Diretórios de resultados
MPICH_DIR = "results/mpich"
SEQUENTIAL_DIR = "results/sequential"
OUTPUT_DIR = "plots"

# Função para ler os tempos dos arquivos
def read_times(file_path):
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            if "real" in line:  # Procura linhas que contêm "real"
                match = re.search(r"real\s+\d+m([\d.]+)s", line)
                if match:
                    times.append(float(match.group(1)))  # Adiciona apenas o tempo em segundos
    return times

# Processa arquivos sequenciais e retorna um dicionário {tamanho: tempo médio}
def process_sequential():
    sequential_results = {}
    for file_name in os.listdir(SEQUENTIAL_DIR):
        if file_name.endswith(".txt"):
            size = int(file_name.split("_")[1].split(".")[0])
            times = read_times(os.path.join(SEQUENTIAL_DIR, file_name))
            sequential_results[size] = {"mean": np.mean(times), "std": np.std(times)}
    return sequential_results

# Processa arquivos MPICH e retorna um dicionário {(tamanho, processos): tempo médio}
def process_mpich():
    mpich_results = {}
    for file_name in os.listdir(MPICH_DIR):
        if file_name.endswith(".txt"):
            parts = file_name.split("_")
            size = int(parts[1])
            procs = int(parts[2].split(".")[0])
            times = read_times(os.path.join(MPICH_DIR, file_name))
            mpich_results[(size, procs)] = {"mean": np.mean(times), "std": np.std(times), "times": times}
    return mpich_results

def plot_sequential_vs_parallel(sequential, mpich_results):
    """
    Gera gráficos separados comparando o tempo sequencial vs paralelo para cada tamanho de imagem,
    incluindo barras de erro com intervalos de confiança.

    Parâmetros:
        sequential: dicionário {tamanho: {"mean": tempo_médio, "std": desvio_padrão}}
        mpich_results: dicionário {(tamanho, processos): {"mean": tempo_médio, "std": desvio_padrão, "times": lista_de_tempos}}
    """
    sizes = sorted(sequential.keys())
    
    for size in sizes:
        # Dados da execução sequencial
        seq_mean = sequential[size]["mean"]
        seq_std = sequential[size]["std"]

        # Dados da execução paralela
        procs = sorted(p for (s, p) in mpich_results.keys() if s == size)
        parallel_means = []
        parallel_stds = []

        for p in procs:
            parallel_mean = mpich_results[(size, p)]["mean"]
            parallel_std = mpich_results[(size, p)]["std"]
            parallel_means.append(parallel_mean)
            parallel_stds.append(parallel_std)

        # Define os valores de erro (barras de erro)
        y_means = [seq_mean] + parallel_means
        y_errors = [seq_std] + parallel_stds

        # Rótulos para o eixo X
        x_labels = ["Sequencial"] + [f"{p} processos" for p in procs]

        # Criação do gráfico com barras de erro
        plt.figure(figsize=(10, 6))
        plt.bar(x_labels, y_means, yerr=y_errors, color="skyblue", capsize=5)
        plt.xlabel("Configuração")
        plt.ylabel("Tempo de execução (s)")
        plt.title(f"Comparação Sequencial vs Paralelo para Tamanho {size}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/sequential_vs_parallel_{size}.png")
        # plt.show()



def plot_speedup(sequential, mpich_results):
    """
    Gera o gráfico de speedup para diferentes tamanhos de imagem e números de processos,
    incluindo barras de erro com intervalos de confiança.
    
    Parâmetros:
        sequential: dicionário {tamanho: {"mean": tempo_médio, "std": desvio_padrão}}
        mpich_results: dicionário {(tamanho, processos): {"mean": tempo_médio, "std": desvio_padrão, "times": lista_de_tempos}}
    """
    sizes = sorted(sequential.keys())
    plt.figure(figsize=(12, 8))

    for size in sizes:
        procs = sorted(p for (s, p) in mpich_results.keys() if s == size)
        sequential_time = sequential[size]["mean"]
        speedups = []
        errors = []

        for p in procs:
            parallel_time = mpich_results[(size, p)]["mean"]
            stddev = mpich_results[(size, p)]["std"]
            speedup = sequential_time / parallel_time
            error = (stddev / parallel_time) * speedup  # Propagação de erro
            speedups.append(speedup)
            errors.append(error)

        plt.errorbar(
            procs,
            speedups,
            yerr=errors,
            fmt='-o',
            capsize=5,
            label=f"Tamanho {size}"
        )

    plt.xlabel("Número de Processos")
    plt.ylabel("Speedup")
    plt.title("Speedup para Diferentes Tamanhos de Imagem")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/speedup_with_confidence.png")
    # plt.show()

def plot_efficiency(mpich_results, fixed_size=1000):
    """
    Gera o gráfico de eficiência para o tamanho fixo de imagem `fixed_size`.
    Apenas plota os valores médios dos tempos paralelos com barras de erro para os diferentes números de processos.

    Parâmetros:
        mpich_results: dicionário {(tamanho, processos): {"mean": tempo_médio, "std": desvio_padrão, "times": lista_de_tempos}}
        fixed_size: tamanho da imagem a ser fixado (default: 1000)
    """
    # Filtra os dados do tamanho fixo
    filtered_results = {p: mpich_results[(fixed_size, p)] for (size, p) in mpich_results if size == fixed_size}

    if not filtered_results:
        print(f"Não há resultados para o tamanho de imagem {fixed_size}.")
        return

    # Extrai os números de processos, médias e desvios padrão
    procs = sorted(filtered_results.keys())
    means = [filtered_results[p]["mean"] for p in procs]
    std_devs = [filtered_results[p]["std"] for p in procs]

    # Criação do gráfico
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        procs, means, yerr=std_devs, fmt="-o", capsize=5, color="black", 
    )
    plt.xlabel("Número de Processos")
    plt.ylabel("Tempo de Execução (s)")
    plt.title(f"Tempo Paralelo para Tamanho de Imagem {fixed_size}")
    plt.xticks(procs)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
    # Salvar o gráfico
    plt.savefig(f"{OUTPUT_DIR}/efficiency_{fixed_size}.png")
    # plt.show()


# Função principal
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sizes = [500, 1000, 5000]

    print("Processando resultados sequenciais...")
    sequential = process_sequential()
    
    print("Processando resultados MPICH...")
    mpich = process_mpich()
    
    print("Gerando gráficos Sequencial vs Paralelo...")
    plot_sequential_vs_parallel(sequential, mpich)
    
    print("Gerando gráficos de Speedup...")
    plot_speedup(sequential, mpich)
    
    print("Gerando gráficos de Eficiência...")
    for size in sizes:
        plot_efficiency(mpich, size)
    
    print(f"Gráficos salvos no diretório '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
