from Perceptron import Perceptron
from Dataset import Dataset
import matplotlib.pyplot as matPlot  # Para visualizacao dos datasets
from matplotlib.colors import ListedColormap  # Lista de cores para plotagens
import pandas as panda  # leitor de arquivo CSV
import statistics
from sklearn import preprocessing

taxa_aprendizado = 1

# Carregar Dataset#1 treino
dados_treinamento_csv = panda.read_csv('Dataset_Treino_1.csv', header=None)
dados_treinamento_csv.head()

# Carregar Dataset#1 teste
dados_teste_csv = panda.read_csv('Dataset_Teste_1.csv', header=None)
dados_teste_csv.head()

# Carregar Dataser#1 teste 1x3
dados_teste_csv_1x3 = panda.read_csv('Dataset_Teste_1_1x3.csv', header=None)
dados_teste_csv_1x3.head()

# Valores da linha 0 a 120 nas posições 0 e 1 são os valores de entrada da função em treinamento
Xtrain = dados_treinamento_csv.iloc[0:120, [0, 1]].values
# Valores da linha 0 a 120 na posição 2 são os valores esperados em treinamento
Ytrain = dados_treinamento_csv.iloc[0:120, 2].values

# Valores da linha 0 a 30 nas posições 0 e 1 são os valores de entrada da função em teste
Xtest = dados_teste_csv.iloc[0:30, [0, 1]].values
# Valores da linha 0 a 30 na posição 2 são os valores esperados em treinamento
Ytest = dados_teste_csv.iloc[0:30, 2].values
########################################################################################################################

# Valores da linha 0 a 30 nas posições 0 e 1 são os valores de entrada da função em teste
X1x2 = dados_teste_csv.iloc[0:20, [0, 1]].values
# Valores da linha 0 a 30 na posição 2 são os valores esperados em treinamento
Y1x2 = dados_teste_csv.iloc[0:20, 2].values
########################################################################################################################

# Valores da linha 0 a 30 nas posições 0 e 1 são os valores de entrada da função em teste
X2x3 = dados_teste_csv.iloc[11:30, [0, 1]].values
# Valores da linha 0 a 30 na posição 2 são os valores esperados em treinamento
Y2x3 = dados_teste_csv.iloc[11:30, 2].values
########################################################################################################################

# Valores da linha 0 a 30 nas posições 0 e 1 são os valores de entrada da função em teste
X1x3 = dados_teste_csv_1x3.iloc[1:20, [0, 1]].values
# Valores da linha 0 a 30 na posição 2 são os valores esperados em treinamento
Y1x3 = dados_teste_csv_1x3.iloc[1:20, 2].values
########################################################################################################################

# Plot dos valores de Treino
cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF22'])
matPlot.figure(figsize=(5, 5))
matPlot.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, cmap=cm_bright)
matPlot.scatter(None, None, color='r', label='Classe 1')
matPlot.scatter(None, None, color='b', label='Classe 2')
matPlot.scatter(None, None, color='g', label='Classe 3')
matPlot.legend()
matPlot.title('Dataset#1 Treino')
matPlot.show()

# Plot dos valores de Teste
cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF22'])
matPlot.figure(figsize=(5, 5))
matPlot.scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest, cmap=cm_bright)
matPlot.scatter(None, None, color='r', label='Classe 1')
matPlot.scatter(None, None, color='b', label='Classe 2')
matPlot.scatter(None, None, color='g', label='Classe 3')
matPlot.legend()
matPlot.title('Dataset#1 Teste')
matPlot.show()

# Plot dos valores de Teste 1x3
cm_bright = ListedColormap(['#FF0000', '#00FF22'])
matPlot.figure(figsize=(5, 5))
matPlot.scatter(X1x3[:, 0], X1x3[:, 1], c=Y1x3, cmap=cm_bright)
matPlot.scatter(None, None, color='r', label='Classe 1')
# matPlot.scatter(None, None, color='b', label='Classe 2')
matPlot.scatter(None, None, color='g', label='Classe 3')
matPlot.legend()
matPlot.title('Dataset#1 Teste 1x3')
matPlot.show()

########################################################################################################################

def main():
#######################################################################################################################
# Perceptron para dupla 1x2
    arquivoD1x2 = open("./dtrain1x2.txt", "r")  # Carrega arquivo D 1x2
    arquivoX1x2 = open("./xtrain1x2.txt", "r")  # Carrega arquivo X 1x2
    dataset1x2 = Dataset(arquivoD1x2, arquivoX1x2)  # Padroniza o Dataset
    teste1x2 = Perceptron(dataset1x2.definindoEntradas(),
                          dataset1x2.definindoValoresDesejados(), taxa_aprendizado)  # Cria uma Perceptron pro dataset 1x2


    teste1x2.treinamento()
    print('')
    print('Resultado do treinamento com Perceptron do Dataset#1 Classes 1 x 2')
    print('w0: {0}  w1: {1}  w2: {2}'.format(teste1x2.getW()[0], teste1x2.getW()[1], teste1x2.getW()[2]))
    print('Número de épocas: {0}'.format(teste1x2.epoca))
    print('ANALISE DO DATASET#1 TESTE PARA AS CLASSES 1X2')
    print('')

    arquivoTesteD1x2 = open("./dtest1x2.txt", "r")  # Carrega arquivo D 1x2
    arquivoTesteX1x2 = open("./xtest1x2.txt", "r")  # Carrega arquivo X 1x2
    datasetTeste1x2 = Dataset(arquivoTesteD1x2, arquivoTesteX1x2)

    teste1x2.predict(datasetTeste1x2.definindoEntradas(),
                 datasetTeste1x2.definindoValoresDesejados(),
                 teste1x2.resultado()[0],
                 teste1x2.resultado()[1])


# Plot dos valores de Teste 1x2

    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    matPlot.figure(figsize=(5, 5))
    matPlot.scatter(X1x2[:, 0], X1x2[:, 1], c=Y1x2, cmap=cm_bright)
    matPlot.scatter(None, None, color='r', label='Classe 1')
    matPlot.scatter(None, None, color='b', label='Classe 2')
    # matPlot.scatter(None, None, color='g', label='Classe 3')
    matPlot.legend()
    matPlot.title('Dataset#1 Teste 1x2')
    matPlot.show()
########################################################################################################################
# Perceptron para dupla 2x3
    arquivoD2x3 = open("./dtrain2x3.txt", "r")  # Carrega arquivo D 2x3
    arquivoX2x3 = open("./xtrain2x3.txt", "r")  # Carrega arquivo X 2x3
    dataset2x3 = Dataset(arquivoD2x3, arquivoX2x3)  # Padroniza o Dataset
    teste2x3 = Perceptron(dataset2x3.definindoEntradas(),
                          dataset2x3.definindoValoresDesejados(),
                          taxa_aprendizado)  # Cria uma Perceptron pro dataset 2x3

    teste2x3.treinamento()

    print('Resultado do treinamento com Perceptron do Dataset#1 Classes 2 x 3')
    print('w0: {0}  w1: {1}  w2: {2}'.format(teste2x3.getW()[0], teste2x3.getW()[1], teste2x3.getW()[2]))
    print('Número de épocas: {0}'.format(teste2x3.epoca))
    print('ANALISE DO DATASET#1 TESTE PARA AS CLASSES 1X2')
    print('')

    arquivoTesteD2x3 = open("./dtest2x3.txt", "r")  # Carrega arquivo D 2x3
    arquivoTesteX2x3 = open("./xtest2x3.txt", "r")  # Carrega arquivo X 2x3
    datasetTeste2x3 = Dataset(arquivoTesteD2x3, arquivoTesteX2x3)

    teste2x3.predict(datasetTeste2x3.definindoEntradas(),
                     datasetTeste2x3.definindoValoresDesejados(),
                     teste2x3.resultado()[0],
                     teste2x3.resultado()[1])


# Plot dos valores de Teste 2x3
    cm_bright = ListedColormap(['#0000FF', '#00FF22'])
    matPlot.figure(figsize=(5, 5))
    matPlot.scatter(X2x3[:, 0], X2x3[:, 1], c=Y2x3, cmap=cm_bright)
    # matPlot.scatter(None, None, color='r', label='Classe 1')
    matPlot.scatter(None, None, color='b', label='Classe 2')
    matPlot.scatter(None, None, color='g', label='Classe 3')
    matPlot.legend()
    matPlot.title('Dataset#1 Teste 2x3')
    matPlot.show()

########################################################################################################################
# Fechamento dos arquivos de dataset#1

    arquivoD1x2.close()
    arquivoX1x2.close()

    arquivoD2x3.close()
    arquivoX2x3.close()

    arquivoTesteX2x3.close()
    arquivoTesteD2x3.close()

main() #execucão da funcão main

