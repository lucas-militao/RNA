import random

class Perceptron:

    def __init__(self, entradas, saidasDesejadas, taxaAprendizagem):
        self.x = entradas
        self.d = saidasDesejadas
        self.w = []
        self.n = taxaAprendizagem
        self.w0 = []
        self.u = []

    #treinamento
    def treinamento(self): #funcao que ira realizar o treinamento

        varW = self.inicializarPesos(len(self.x[0]))
        varW0 = self.inicializarLimiar()
        varU = 0

        epoca = 0

        y = 0 #variável que irá receber o sinal(u)

        for i in range(len(self.x)):

            erro = False

            while (erro != True):

                # self.u.append(self.calculoSaida(self.x[i], varW, varW0))
                varU = self.calculoSaida(self.x[i], varW, varW0)

                y = 1 if varU >= 0 else -1

                if(y != self.d[i]):
                    varW = self.atualizarPesos(varW, self.n, (self.d[i] - y), self.x[i])
                    varW0 = self.atualizarLimiar(varW0, self.n, (self.d[i] - y))

                else:
                    self.u.append(varU)
                    self.w.append(varW)
                    self.w0.append(varW0)
                    erro = True

                epoca += 1
    #----------------------------------------------------------------------

    #cálculo da saída (u)
    def calculoSaida(self, x, w, w0):

        resultado = 0

        for i in range(len(x)):
            resultado += x[i]*w[i]
        resultado -= w0

        return resultado
    #------------------------------------------------------------

    #método padrão para cálculo dos novos pesos
    def atualizarPesos(self, w, n, erro, x): #funcao que irá atualizar os pesos
        resultado = []

        for i in range(len(w)):
            resultado.append(
                w[i] + n*erro*x[i]
            )

        return resultado

    def atualizarLimiar(self, w0, n, erro): #funcao que irá atualizar o limiar
        return w0 + n*erro*(-1)
    #-------------------------------------------------------------------------

    #inicializar pesos e limiar com dados aleatórios
    def inicializarPesos(self, n):
        w = []

        for i in range(n):
            w.append(random.random())
        return w

    def inicializarLimiar(self):
        return random.random()
    #-----------------------------------------------

    def resultado(self):
        print(self.w)

