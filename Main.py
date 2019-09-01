from Perceptron import Perceptron
from Dataset import Dataset

#valores do exercício para teste
entradas = [[0.5, 1.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.3], [1.5, 0.5]]
saidasDesejadas = [1, -1, 1, -1, 1]
pesosSinapticos = [[1, 2]]
taxaAprendizagem = 0.1
limiar = [-2]

#útil
# with open('./dtest.txt') as f:
#     conteudo = f.readlines()

def main():

    arquivoD = open("./dtest.txt", "r")
    arquivoX = open("./xtest.txt", "r")

    dataset = Dataset(arquivoD, arquivoX)

    # print(dataset.definindoValoresDesejados())
    print(dataset.definindoEntradas())

    arquivoD.close()

    #teste com os valores do exercício
    # teste = Perceptron(entradas, saidasDesejadas, pesosSinapticos, taxaAprendizagem, limiar)
    #
    # teste.treinamento()
    #
    # print(teste.w)


main()

