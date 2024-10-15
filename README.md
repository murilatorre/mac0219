## TO-DO

- [x] Paralelizar o código com PThreads
- [x]  Paralelizar o código com OpenMP
- [x]  Revisar os códigos e merge
- [x]  Realizar a simulação dos testes e coletar os dados
- [ ]  Os gráficos têm como título 'Média e Intervalo de Confiança ...', mas esses não estão sendo calculados (não sei a razão). Devemos decidir entre renomeá-los ou consertar.
- [ ]  Escrever o relatório
- [ ]  Entregar


## Visualizando os resultados

Para criar os gráficos e arquivos .csv siga as instruções abaixo:
1. Rode `cd views/` para mudar de diretório,
2. Rode `./run.sh` para preparar o ambiente virtual,
3. Ative o ambiente virtual `source ./.venv/bin/activate`.
   
Agora, podemos gerar o arquivo .csv pela linha de comando:
```
  python3 extract_perf_data.py
```
Em seguida, para gerar os gráficos, devemos rodar:
```
  python3 create_performance_graphs.py
```

O arquivo .csv é nomeado 'perf_stats.csv' e as imagens geradas estarão na pasta `views/graphs/`.

NOTA: Só é necessário preparar o ambiente virutal uma vez, após a primeira vez ignore o passo 2. Além disso, rode `deactivate` para desativar o ambiente virtual.

### Informações sobre os resultados

* O tamanho da entrada varia entre $$2^{4}$$ e $$2^{11}$$
* O número threads varia entre $$2^{0}$$ e $$2^{5}$$
* As simulações que variam o tamanho da entrada usam o número de threads default 16;
* As simulações que variam o número de threads usam o tamanho de entrada default 512;
