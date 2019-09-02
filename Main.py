from Perceptron import Perceptron
from Dataset import Dataset

#valores do exercício para teste
# entradas = [[0.5, 1.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.3], [1.5, 0.5]]
# saidasDesejadas = [1, -1, 1, -1, 1]
# pesosSinapticos = [[1, 2]]
# taxaAprendizagem = 0.1
# limiar = [-2]

def main():

    arquivoD = open("./dtest.txt", "r")
    arquivoX = open("./xtest.txt", "r")

    dataset = Dataset(arquivoD, arquivoX)

    teste = Perceptron(dataset.definindoEntradas(), dataset.definindoValoresDesejados(), 1)

    teste.treinamento()

    arquivoD.close()

    for i in teste.u:
        print(i)

main()

