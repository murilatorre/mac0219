## TO-DO

- [x] Tarefa 1 - Versão Sequencial (sequential_julia.c + Makefile)

- [x] Tarefa 2 - Versão Paralela (1D_parallel_julia.c. + Makefile)

	- [x] Primeira Parte - Distribuição dos Dados

	- [x] Segunda Parte - Computação do Conjunto Usando MPI
		
	- [x] Terceira Parte - Salvando num Arquivo

- [ ] Tarefa 3 - Experimentos

- [ ] Entregar

## Pré-Requisitos

Para rodar o exercício programa na sua máquina local, você precisa ter as seguintes ferramentas instaladas:
  
1. Instalar o SimGrid
	```
    sudo apt update && apt upgrade -y
	sudo apt install simgrid
	```

2. Instalar uma implementação de MPI. Abaixo segue instruções para instalar OpenMPI (minha recomendação)
	```
	sudo apt update && apt upgrade
	sudo apt install --reinstall openmpi-bin openmpi-common libopenmpi-dev
	```
