import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import sem

# Função para calcular a média e o intervalo de confiança
def compute_statistics(df, metric, groupby):
    grouped = df.groupby(['version', 'region', groupby])[metric]
    stats = grouped.agg(['mean', sem]).reset_index()
    stats['ci95_hi'] = stats['mean'] + 1.96 * stats['sem']
    stats['ci95_lo'] = stats['mean'] - 1.96 * stats['sem']
    return stats

# Função para criar gráficos para uma métrica específica
def plot_metric(stats, metric):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=stats, x='region', y='mean', hue='version', errorbar=None, palette='muted')
    
    # Adiciona os intervalos de confiança
    for index, row in stats.iterrows():
        plt.errorbar(x=row['region'], y=row['mean'],
                     yerr=[[row['mean'] - row['ci95_lo']], [row['ci95_hi'] - row['mean']]],
                     fmt='none', c='black', capsize=5)
    
    plt.title(f'Média e Intervalo de Confiança (95%) para {metric}')
    plt.xlabel('Região')
    plt.ylabel(metric)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Versão', loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Cria a pasta de gráficos se não existir
    os.makedirs('graphs', exist_ok=True)
    # Salva o gráfico como uma imagem na pasta graphs
    plt.savefig(f'graphs/{metric}_comparison.png')
    plt.close()

# Função para criar gráficos para uma métrica específica em relação ao tamanho da entrada
def plot_metric_input_size_comparision(stats, metric):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stats, x='input_size', y='mean', hue='version', style='region', markers=True, errorbar=None, palette='muted')
    
    # Adiciona os intervalos de confiança
    for index, row in stats.iterrows():
        plt.errorbar(x=row['input_size'], y=row['mean'],
                     yerr=[[row['mean'] - row['ci95_lo']], [row['ci95_hi'] - row['mean']]],
                     fmt='none', c='black', capsize=5)
    
    plt.title(f'Média e Intervalo de Confiança (95%) para {metric} por Tamanho de Entrada')
    plt.xlabel('Tamanho da Entrada')
    plt.xscale('log',base=2) 
    plt.ylabel(metric)
    plt.yscale('log') 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Versão e Região', loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Cria a pasta de gráficos se não existir
    os.makedirs('graphs', exist_ok=True)
    # Salva o gráfico como uma imagem na pasta graphs
    plt.savefig(f'graphs/{metric}_input_size_comparison.png')
    plt.close()


# Função para criar gráficos para uma métrica específica em relação ao número de threads
def plot_metric_num_threads_comparision(stats, metric):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stats, x='threads', y='mean', hue='version', style='region', markers=True, errorbar=None, palette='muted')
    
    # Adiciona os intervalos de confiança
    for index, row in stats.iterrows():
        plt.errorbar(x=row['threads'], y=row['mean'],
                     yerr=[[row['mean'] - row['ci95_lo']], [row['ci95_hi'] - row['mean']]],
                     fmt='none', c='black', capsize=5)
    
    plt.title(f'Média e Intervalo de Confiança (95%) para {metric} por Número de Threads')
    plt.xlabel('Número de Threads')
    plt.xscale('log',base=2) 
    plt.ylabel(metric)
    plt.yscale('log') 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Versão e Região', loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Cria a pasta de gráficos se não existir
    os.makedirs('graphs', exist_ok=True)
    # Salva o gráfico como uma imagem na pasta graphs
    plt.savefig(f'graphs/{metric}_num_threads_comparison.png')
    plt.close()



# Carregar os dados do CSV
df = pd.read_csv('perf_stats.csv')

# Calcular estatísticas para cada métrica e gerar os gráficos
# Adicione as métricas conforme presente no log do perf
metrics = ['time', 'cycles', 'instructions', 'branches', 'branch_misses', 'context_switches']
for metric in metrics:
    stats = compute_statistics(df, metric, 'input_size')
    plot_metric(stats, metric)
    plot_metric_input_size_comparision(stats, metric)


# Carregar os dados do CSV do número de threads
df = pd.read_csv('perf_stats_threads.csv')
for metric in metrics:
    stats = compute_statistics(df, metric, 'threads')
    plot_metric_num_threads_comparision(stats, metric)


print("Gráficos gerados com sucesso e salvos na pasta 'graphs'!")
