import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import sem

# Função para calcular a média e o intervalo de confiança
def compute_statistics(df, metric, groupby):
    grouped = df.groupby(groupby)[metric]
    stats = grouped.agg(['mean', sem]).reset_index()
    stats['ci95_hi'] = stats['mean'] + 1.96 * stats['sem']
    stats['ci95_lo'] = stats['mean'] - 1.96 * stats['sem']
    return stats


# Função para criar gráficos para uma métrica específica em relação a região
def plot_metric_region_comparison(df, metric):
    stats = compute_statistics(df, metric, ['version', 'region'])

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(data=stats, x='region', y='mean', hue='version', palette='muted', errorbar=None)
    
    # Adiciona os intervalos de confiança para cada versão e região
    for index, row in stats.iterrows():
        region_index = stats['region'].unique().tolist().index(row['region'])
        version_offset = {'mandelbrot_omp': -0.25, 'mandelbrot_pth': 0, 'mandelbrot_seq': 0.25}
        x_position = region_index + version_offset[row['version']]
        
        plt.errorbar(
            x=x_position, 
            y=row['mean'],
            yerr=[[row['mean'] - row['ci95_lo']], [row['ci95_hi'] - row['mean']]],
            fmt='none', 
            c='black', 
            capsize=5, 
            label='_nolegend_'
        )

    plt.title(f'Média e Intervalo de Confiança (95%) para {metric}')
    plt.xlabel('Região')
    plt.ylabel(metric)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Versão', loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Cria a pasta de gráficos se não existir
    os.makedirs('graphs/regions', exist_ok=True)
    plt.savefig(f'graphs/regions/{metric}_comparison.png')
    plt.close()

# Função para criar gráficos para uma métrica específica em relação ao tamanho da entrada
def plot_metric_input_size_comparison(df, metric):
    stats = compute_statistics(df, metric, ['version', 'input_size'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stats, x='input_size', y='mean', hue='version', markers=True, errorbar=None, palette='muted')
    
    # Adiciona os intervalos de confiança, com deslocamento e cores diferenciadas
    colors = sns.color_palette('muted', n_colors=len(stats['version'].unique())) 
    for idx, (version, color) in enumerate(zip(stats['version'].unique(), colors)):
        version_data = stats[stats['version'] == version]
        
        for i, row in version_data.iterrows():
            plt.errorbar(x=row['input_size'], y=row['mean'],
                         yerr=[[row['mean'] - row['ci95_lo']], [row['ci95_hi'] - row['mean']]],
                         fmt='none', ecolor=color, capsize=6, alpha=0.8)
    
    plt.title(f'Média e Intervalo de Confiança (95%) para {metric} por Tamanho de Entrada')
    plt.xlabel('Tamanho da Entrada')
    plt.xscale('log', base=2) 
    plt.ylabel(metric)
    plt.yscale('log') 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Versão', loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Cria a pasta de gráficos se não existir
    os.makedirs('graphs/input_size', exist_ok=True)
    plt.savefig(f'graphs/input_size/{metric}_input_size_comparison.png')
    plt.close()

# Função para criar gráficos para uma métrica específica em relação ao número de threads
def plot_metric_num_threads_comparison(df, metric):
    stats = compute_statistics(df, metric, ['version', 'threads'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=stats, x='threads', y='mean', hue='version', markers=True, errorbar=None, palette='muted')
   
    # Adiciona os intervalos de confiança, com deslocamento e cores diferenciadas
    colors = sns.color_palette('muted', n_colors=len(stats['version'].unique()))
    for idx, (version, color) in enumerate(zip(stats['version'].unique(), colors)):
        version_data = stats[stats['version'] == version]
        
        for i, row in version_data.iterrows():
            plt.errorbar(x=row['threads'], y=row['mean'],
                         yerr=[[row['mean'] - row['ci95_lo']], [row['ci95_hi'] - row['mean']]],
                         fmt='none', ecolor=color, capsize=6, alpha=0.8)


    plt.title(f'Média e Intervalo de Confiança (95%) para {metric} por Número de Threads')
    plt.xlabel('Número de Threads')
    plt.xscale('log',base=2) 
    plt.ylabel(metric)
    plt.yscale('log') 
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Versão', loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Cria a pasta de gráficos se não existir
    os.makedirs('graphs/num_threads', exist_ok=True)
    plt.savefig(f'graphs/num_threads/{metric}_num_threads_comparison.png')
    plt.close()

def main():
    os.makedirs('graphs', exist_ok=True)
    
    # Carregar os dados do CSV
    df = pd.read_csv('perf_stats.csv')
    threads_df = pd.read_csv('perf_stats_threads.csv')

    # Calcular estatísticas para cada métrica e gerar os gráficos
    # Adicione as métricas conforme presente no log do perf
    metrics = ['time', 'cycles', 'instructions', 'branches', 'branch_misses', 'context_switches']
    for metric in metrics:
        plot_metric_region_comparison(df, metric)
        plot_metric_input_size_comparison(df, metric)
        plot_metric_num_threads_comparison(threads_df, metric)


    print("Gráficos gerados com sucesso e salvos na pasta 'graphs'!")

main()