import random

class Perceptron:

    def __init__(self, entradas, saidasDesejadas, taxaAprendizagem):
        self.x = entradas
        self.d = saidasDesejadas
        self.w = []
        self.n = taxaAprendizagem
        self.w0 = []
        self.u = []

    def treinamento(self): #funcao que ira realizar o treinamento

        varW = self.inicializarPesos()
        varW0 = 0

        #erro que irá relatar se o resultado foi o esperado
        #epoca numero de interacoes
        #atualizacoes numero de atualizacoes dos pesos

        erro = False
        epoca = 0
        atualizacoes = 0


        y = 0 #variável que irá receber o sinal(u)

        for i in range(len(self.x)):

            erro = False

            while (erro != True):

                self.u.append(self.calculoSaida(self.x[i], varW, varW0))

                y = 1 if self.u[epoca] >= 0 else -1

                if(y != self.d[i]):
                    # self.w.append(self.atualizarPesos(self.w[atualizacoes], self.n, (self.d[i] - y), self.x[i]))
                    # self.w0.append(self.autalizarLimiar(self.w0[atualizacoes], self.n, (self.d[i] - y)))
                    # atualizacoes += 1
                    varW = self.atualizarPesos(len(self.x[i]))
                    varW0 = self.atualizarLimiar()

                else:
                    self.w.append(varW)
                    self.w0.append(varW0)
                    erro = True

                epoca += 1

    def calculoSaida(self, x, w, w0): #funcao ira calcular o U

        resultado = 0

        for i in range(len(x)):
            resultado += x[i]*w[i]
        resultado -= w0

        return resultado

    #método padrão para cálculo dos novos pesos
    # def atualizarPesos(self, w, n, erro, x): #funcao que irá atualizar os pesos
    #     resultado = []
    #
    #     for i in range(len(w)):
    #         resultado.append(
    #             w[i] + n*erro*x[i]
    #         )
    #
    #     return resultado

    # def autalizarLimiar(self, w0, n, erro):
    #     return w0 + n*erro*(-1)

    def atualizarPesos(self, n):
        w = []

        for i in range(n):
            w.append(random.random())
        return w

    def atualizarLimiar(self):
        return random.random()

    def inicializarPesos(self):
        w = []
        for i in range(len(self.x[0])):
            w.append(0)

        return w


    def resultado(self):
        print(self.w)

