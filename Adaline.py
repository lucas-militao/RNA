import random
import pandas as pd
import statistics
from sklearn import preprocessing

class Adaline:

    def __init__(self, entradas, saidasDesejadas, taxaAprendizagem, precisao_requerida):
        # self.normalization(entradas)

        self.x = entradas
        self.d = saidasDesejadas
        self.w = []
        self.n = taxaAprendizagem
        self.w0 = []
        self.epoca = 0
        self.Eqm = []
        self.Eqm_anterior = 100
        self.Eqm_atual = 0
        self.precisao = precisao_requerida

        # treinamento
    def treinamento_online(self):  # funcao que ira realizar o treinamento
        # self.varW = self.inicializarPesos(len(self.x[0]))
        # self.varW0 = self.inicializarLimiar()
        self.w.append(self.inicializarPesos(len(self.x[0])))
        self.w0.append(self.inicializarLimiar())
        # self.w = [[0,0]]
        # self.w0 = [0]
        self.varU = 0
        self.u = []
        self.atualizacao = 0
        y = 0  # variável que irá receber o sinal(u)

        while abs(self.Eqm_atual - self.Eqm_anterior) >= self.precisao:
            self.Eqm_anterior = self.Eqm_atual
            # self.Eqm.append(self.Eqm_anterior)

            for i in range(len(self.x)):
                self.varU = self.calculoSaida(self.x[i], self.w[i], self.w0[i])
                self.u.append(self.varU)
                self.w.append(self.atualizarPesos(self.w[i], self.n, (self.d[i] - self.varU), self.x[i]))
                self.w0.append(self.atualizarLimiar(self.w0[i], self.n, (self.d[i] - self.varU)))

            self.Eqm_atual = self.calcEqm(len(self.x), self.u, self.d)
            self.Eqm.append(self.Eqm_atual)
            self.epoca += 1
    # ----------------------------------------------------------------------

    # cálculo do Eqm
    def calcEqm(self, p, u, d):

        resultado = 0

        for i in range(p):
            resultado += pow(d[i] - u[i], 2)
        resultado = resultado/p

        return resultado

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

    # inicializar pesos e limiar com dados nulos
    def inicializarPesosNulos(self, n):
        w = []

        for i in range(n):
            w.append(random.random())
        return w

    def inicializarLimiarNulos(self):
        return random.random()
    #------------------------------------------------

    def resultado(self):
        return self.w[len(self.w) - 1]

    def getW(self):
        w = [self.w0[len(self.w0) - 1],
             self.w[len(self.w) - 1][0],
             self.w[len(self.w) - 1][1],
             self.w[len(self.w) - 1][2],
             self.w[len(self.w) - 1][3]]
        return w

    def normalization(self, x):

        for i in range(len(x)):
            for j in x[i]:
                x[i][j] = (j - statistics.mean(x[i]))/statistics.pstdev(x[i])

        return x


    def predict(self, entradas, valoresDesejado, w):

        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        w3 = w[3]
        w4 = w[4]

        for i in range(len(entradas)):
            print('x1: {0}  x2: {1} x3: {2}  x4: {3} D: {4}'.format(entradas[i][0], entradas[i][1], entradas[i][2], entradas[i][3], valoresDesejado[i]))
            resultado = entradas[i][0]*w1 + entradas[i][1]*w2 + entradas[i][2]*w3 + entradas[i][3]*w4 - w0
            print('Resultado antes da operação: {0}'.format(resultado))
            if resultado >= 0:
                resultado = 1
            elif resultado < 0:
                resultado = -1

            print('Iteração: {0} Resultado esperado: {1}  Resultado obtido: {2}'.format(i+1, valoresDesejado[i], resultado))
            print('-------------------------------------------------------------------------------------------------------')

        print('Eqm ======= {0}'.format(self.Eqm))

    def conferirRespostas(self, x, w, w0, d, p): #imprime a porcentagem de acertos

         varU = []
         varY = []
         resultado = 0
         acertos = 0

         for i in range(p):
             for j in range(len(w[i])):
                 resultado += w[i][j]*x[i][j]
             resultado -= w0[i]
             varU.append(resultado)
             varY.append(1 if resultado >= 0 else -1)
             print('valor resultante: {0}\tvalor desejado: {1}'.format(varY[i], d[i]))

         for k in range(p):
             if(varY[k] == d[k]):
                 acertos += 1

         print('{0}% de acertos'.format((acertos*100)/p))

