from Perceptron import Perceptron
from Dataset import Dataset
from Adaline import Adaline
import matplotlib.pyplot as matPlot  # Para visualizacao dos datasets
from matplotlib.colors import ListedColormap  # Lista de cores para plotagens
import pandas as panda  # leitor de arquivo CSV
import statistics
from sklearn import preprocessing

taxa_aprendizado = 0.000001
precisao = pow(10, -5)

########################################################################################################################

def mainAdaline():

    arquivoDTest = open("Dataset2/dtest2.txt", "r")
    arquivoXTest = open("Dataset2/xtest2.txt", "r")
    datasetTest = Dataset(arquivoDTest, arquivoXTest)

    arquivoDTrain = open("Dataset2/dtrain2.txt", "r")
    arquivoXTrain = open("Dataset2/xtrain2.txt", "r")
    datasetTrain = Dataset(arquivoDTrain, arquivoXTrain)

    redeAdalineTrain = Adaline(datasetTrain.definindoEntradas(), datasetTrain.definindoValoresDesejados(), taxa_aprendizado, precisao)

    redeAdalineAnd = Adaline([[1,1], [0,0], [1,0], [0,1]], [1,0,0,0], 0.0025, 0.0001)

    redeAdalineAnd.treinamento_online()

    print(redeAdalineAnd.epoca)

    # redeAdalineTrain.treinamento_online()
    #
    # print('')
    # print('Resultado do treinamento com Perceptron do Dataset#1 Classes 1 x 2')
    # print('w0: {0}  w1: {1}  w2: {2} w3: {3} w4: {4}'.format(redeAdalineTrain.getW()[0], redeAdalineTrain.getW()[1], redeAdalineTrain.getW()[2], redeAdalineTrain.getW()[3], redeAdalineTrain.getW()[4]))
    # print('Número de épocas: {0}'.format(redeAdalineTrain.epoca))
    # print('ANALISE DO DATASET#1 TESTE PARA AADALINE')
    # print('')
    #
    # redeAdalineTrain.predict(datasetTest.definindoEntradas(),
    #                  datasetTest.definindoValoresDesejados(),
    #                  redeAdalineTrain.getW())







#######################################################################################################################


mainAdaline() #execucão da funcão main

