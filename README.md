## TO-DO

- [x] Paralelizar o código com PThreads
- [x]  Paralelizar o código com OpenMP
- [ ]  Revisar os códigos e merge
- [ ]  Realizar a simulação dos testes e coletar os dados
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

NOTA: Só é necessário preparar o ambiente virutal uma vez, após a primeira vez comece pelo passo 3. Além disso, rode `deactivate` para desativar o ambiente virtual.
