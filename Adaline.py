import random
import pandas as pd
import statistics
from sklearn import preprocessing

class Adaline:

    def __init__(self, entradas, saidasDesejadas, taxaAprendizagem, precisao_requerida):
        # entradas = preprocessing.normalize(entradas) #dados normalizados

        self.x = entradas
        self.d = saidasDesejadas
        self.w = []
        self.n = taxaAprendizagem
        self.w0 = []
        self.epoca = 0
        self.Eqm = []
        self.Eqm_anterior = 0
        self.Eqm_atual = 0
        self.precisao = precisao_requerida

        # treinamento
    def treinamento_online(self):  # funcao que ira realizar o treinamento
        # self.varW = self.inicializarPesos(len(self.x[0]))
        # self.varW0 = self.inicializarLimiar()
        self.w.append(self.inicializarPesos(len(self.x[0])))
        self.w0 = self.inicializarLimiar()
        self.varU = 0
        self.u = []
        self.atualizacao = 0
        y = 0  # variável que irá receber o sinal(u)

        while self.Eqm_atual - self.Eqm_anterior <= self.precisao:
            self.Eqm_anterior = self.calcEqm()
            self.Eqm.append(self.Eqm_anterior)

            for i in range(len(self.x)):
                self.varU = self.calculoSaida(self.x[i], self.w[self.atualizacao])
                self.u.append(self.varU)
                self.w.append(self.atualizarPesos(self.w[self.atualizacao], self.n, (self.d[i] - self.varU), self.x[i]))
                self.w0.append(self.atualizarLimiar(self.w0[self.atualizacao], self.n, (self.d[i] - self.varU)))

            self.Eqm_atual = self.calcEqm(len(self.x))
            self.Eqm.append(self.Eqm_atual)
            self.epoca += 1
    # ----------------------------------------------------------------------

    # cálculo do Eqm
    def calcEqm(self, p, u):

        self.varU = 0
        self.resultado = 0

        for i in range(p):
            self.resultado += u[i]

        return self.resultado

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

    def normalization(self, x):

        for i in range(len(x)):
            for j in x[i]:
                x[i][j] = (j - statistics.mean(x[i]))/statistics.pstdev(x[i])

        return x


    def predict(self, entradas, valoresDesejado, w1, w2):
        # normalizar = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(entradas)
        # entradas = normalizar.transform(entradas)
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