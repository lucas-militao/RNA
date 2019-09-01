from Perceptron import Perceptron

entradas = [[0.5, 1.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.3], [1.5, 0.5]]
saidasDesejadas = [1, -1, 1, -1, 1]
pesosSinapticos = [[1, 2]]
taxaAprendizagem = 0.1
limiar = [-2]

def main():

    teste = Perceptron(entradas, saidasDesejadas, pesosSinapticos, taxaAprendizagem, limiar)

    teste.treinamento()

    print(teste.w)


main()

