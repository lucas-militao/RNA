import random
from sklearn import preprocessing

class Perceptron:

    def __init__(self, entradas, saidasDesejadas, taxaAprendizagem):
        normalizar = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(entradas)
        entradas = normalizar.transform(entradas)

        self.x = entradas
        self.d = saidasDesejadas
        self.w = []
        self.n = taxaAprendizagem
        self.w0 = []
        self.u = []
        self.epoca = 0

        # treinamento
    def treinamento(self):  # funcao que ira realizar o treinamento
        varW = self.inicializarPesos(len(self.x[0]))
        varW0 = self.inicializarLimiar()
        varU = 0
        y = 0  # variável que irá receber o sinal(u)

        erro = True
        while erro:

            erro = False

            for i in range(len(self.x)):
                varU = self.calculoSaida(self.x[i], varW, varW0)

                y = 1 if varU >= 0 else -1

                if (y != self.d[i]):
                    varW = self.atualizarPesos(varW, self.n, (self.d[i] - y), self.x[i])
                    varW0 = self.atualizarLimiar(varW0, self.n, (self.d[i] - y))
                    self.w.append(varW)
                    self.w0.append(varW0)
                    erro = True
                else:
                    self.u.append(varU)
                    self.w.append(varW)
                    self.w0.append(varW0)
            self.epoca += 1
    # ----------------------------------------------------------------------

    # cálculo da saída (u)
    def calculoSaida(self, x, w, w0):

        resultado = 0

        for i in range(len(x)):
            resultado += x[i] * w[i]
        resultado -= w0

        return resultado
    # ------------------------------------------------------------

    # método padrão para cálculo dos novos pesos
    def atualizarPesos(self, w, n, erro, x):  # funcao que irá atualizar os pesos
        resultado = []

        for i in range(len(w)):
            resultado.append(
                w[i] + n * erro * x[i]
            )

        return resultado

    def atualizarLimiar(self, w0, n, erro):  # funcao que irá atualizar o limiar
        return w0 + n * erro * (-1)
    # -------------------------------------------------------------------------

    # inicializar pesos e limiar com dados aleatórios
    def inicializarPesos(self, n):
        w = []

        for i in range(n):
            w.append(random.random())
        return w

    def inicializarLimiar(self):
        return random.random()
    # -----------------------------------------------

    def resultado(self):
        return self.w[len(self.w) - 1]

    def getW(self):
        w = [self.w0[len(self.w0) - 1],
             self.w[len(self.w) - 1][0],
             self.w[len(self.w) - 1][1]]
        return w

    def predict(self, entradas, valoresDesejado, w1, w2):
        normalizar = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(entradas)
        entradas = normalizar.transform(entradas)
        w0 = self.w0[len(self.w0)-1]

        for i in range(len(entradas)):
            print('x1: {0}  x2: {1} D: {2}'.format(entradas[i][0], entradas[i][1], valoresDesejado[i]))
            resultado = entradas[i][0]*w1 + entradas[i][1]*w2 - w0
            print('Resultado antes da operação: {0}'.format(resultado))
            if resultado >= 0:
                resultado = 1
            elif resultado < 0:
                resultado = -1

            print('Iteração: {0} Resultado esperado: {1}  Resultado obtido: {2}'.format(i+1, valoresDesejado[i], resultado))
            print('-------------------------------------------------------------------------------------------------------')