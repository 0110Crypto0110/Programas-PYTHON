import time
import psutil  
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------
# Bubble_sort
#-------------------------------------------------------------
# Serve para ordenar os elementos de uma lista, levando o maior valor para o final da lista
#-------------------------------------------------------------
# funcoes adicionais para mensurar tempo e uso de memoria
#-------------------------------------------------------------
def bubble_sort(arr):
    inicio = time.time()

    # Monitorando o uso de memória antes da execução
    memoria_inicial = psutil.Process().memory_info().rss

    for n in range(len(arr) - 1, 0, -1):
        troca = False
        for i in range(n):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                troca = True
        if not troca:
            break

    fim = time.time()
    tempo_execucao = fim - inicio

    # Monitorando o uso de memória após a execução
    memoria_final = psutil.Process().memory_info().rss
    memoria_usada = memoria_inicial - memoria_final 

    return arr, memoria_usada, tempo_execucao

#-------------------------------------------------------------
# Ler e converter arquivo para array
#-------------------------------------------------------------
# Função para ler um arquivo de texto e converter as linhas em um array de inteiros
#-------------------------------------------------------------
def ler_arquivo_e_converter_para_array(caminho_arquivo_entrada):
    with open(caminho_arquivo_entrada, 'r') as file:
        linhas = file.readlines()

    # Remover espaços em branco e converter para inteiros
    arr = [int(linha.strip()) for linha in linhas]

    return arr

#-------------------------------------------------------------
# Escrever array em arquivo
#-------------------------------------------------------------
# Função para escrever um array em um arquivo .txt
#-------------------------------------------------------------
def escrever_array_em_arquivo(arr, caminho_arquivo_saida):
    with open(caminho_arquivo_saida, 'w') as arquivo:
        arquivo.write(f"Lista ordenada: {arr}\n")

#-------------------------------------------------------------
# Escrever dados de memória e tempo em arquivo
#-------------------------------------------------------------
# Função para escrever os dados de memória e tempo em um arquivo .txt
#-------------------------------------------------------------
def escrever_dados_em_arquivo(memoria_usada, tempo_execucao, caminho_arquivo_saida):
    with open(caminho_arquivo_saida, 'a') as arquivo:  # Abrir em modo append para adicionar sem sobrescrever
        # Memória usada
        # Tempo de execução
        arquivo.write(f"{memoria_usada}\n")
        arquivo.write(f"{tempo_execucao}\n")

# Caminhos dos arquivos
caminho_arquivo_entrada = 'arq.txt'
caminho_arquivo_saida_ARRAY = 'arq-saidaARRAY.txt'
caminho_arquivo_saida_MEDIA = 'arq-saidaMEDIA.txt'

# Ler o array do arquivo uma vez
arr_inicial = ler_arquivo_e_converter_para_array(caminho_arquivo_entrada)

for i in range(10):
    # Criar uma cópia do array para evitar ordenar o mesmo array várias vezes
    arr = arr_inicial[:]

    # Ordenar o array e obter o uso de memória e tempo de execução
    arr_ordenado, memoria_usada, tempo_execucao = bubble_sort(arr)

    # Escrever a lista ordenada apenas na primeira iteração
    if i == 0:
        escrever_array_em_arquivo(arr_ordenado, caminho_arquivo_saida_ARRAY)

    # Escrever os dados de memória e tempo para cada iteração no arquivo
    escrever_dados_em_arquivo(memoria_usada, tempo_execucao, caminho_arquivo_saida_MEDIA)

#-------------------------------------------------------------
# Funções para leitura e processamento de linhas do arquivo
#-------------------------------------------------------------

# Função para ler as linhas de um arquivo
def ler_linhas(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

# Função para obter valores das linhas pares
def obter_valores_linhas_pares(linhas):
    return [float(linhas[i].strip()) for i in range(len(linhas)) if i % 2 == 0]

# Função para obter valores das linhas ímpares
def obter_valores_linhas_impares(linhas):
    return [float(linhas[i].strip()) for i in range(len(linhas)) if i % 2 != 0]

# Função para calcular média e mediana de uma lista de valores
def calcular_media_e_mediana(valores):
    media = np.mean(valores)
    mediana = np.median(valores)
    return media, mediana

#-------------------------------------------------------------
# Gerar gráficos de pontos
#-------------------------------------------------------------
# Função para gerar gráficos de pontos com valores de memória e tempo de execução
#-------------------------------------------------------------
def gerar_graficos(memoria_usada, tempo_execucao):
    # Gerar gráfico de pontos
    plt.figure()
    plt.scatter(range(len(memoria_usada)), memoria_usada, color='blue', label='Memória Usada (Bytes)')
    plt.scatter(range(len(tempo_execucao)), tempo_execucao, color='red', label='Tempo de Execução (Segundos)')
    plt.legend()
    plt.title('Uso de Memória e Tempo de Execução')
    plt.xlabel('Execuções')
    plt.ylabel('Valores')
    plt.show()

#-------------------------------------------------------------
# Função principal
#-------------------------------------------------------------
# Função principal para ler dados, processar e gerar gráficos
#-------------------------------------------------------------
def main():
    filename = 'arq-saidaMEDIA.txt'
    linhas = ler_linhas(filename)

    memoria_usada = obter_valores_linhas_impares(linhas)
    tempo_execucao = obter_valores_linhas_pares(linhas)

    gerar_graficos(memoria_usada, tempo_execucao)

    media_memoria, mediana_memoria = calcular_media_e_mediana(memoria_usada)
    media_tempo, mediana_tempo = calcular_media_e_mediana(tempo_execucao)

    print(f'Média da memória usada: {media_memoria:.2f} Bytes')
    print(f'Mediana da memória usada: {mediana_memoria:.2f} Bytes')
    print(f'Média do tempo de execução: {media_tempo:.2f} Segundos')
    print(f'Mediana do tempo de execução: {mediana_tempo:.2f} Segundos')

if __name__ == "__main__":
    main()
