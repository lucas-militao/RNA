class Perceptron:

    def __init__(self, entradas, saidasDesejadas, pesosSinapticos, taxaAprendizagem, limiar):
        self.x = entradas
        self.d = saidasDesejadas
        self.w = pesosSinapticos
        self.n = taxaAprendizagem
        self.w0 = limiar
        self.u = []

    def treinamento(self): #funcao que ira realizar o treinamento

        #erro que irá relatar se o resultado foi o esperado
        #epoca numero de interacoes
        #atualizacoes numero de atualizacoes dos pesos
        #o contador

        erro = False
        epoca = 0
        atualizacoes = 0
        count = 0

        y = 0 #variável que irá receber o sinal(u)

        for i in range(len(self.x)):

            erro = False

            while (erro != True):

                self.u.append(self.calculoSaida(self.x[i], self.w[atualizacoes], self.w0[atualizacoes]))

                y = 1 if self.u[epoca] >= 0 else -1

                if(y != self.d[epoca]):
                    self.w.append(self.atualizarPesos(self.w[atualizacoes], self.n, (self.d[i] - y), self.x[i]))
                    self.w0.append(self.autalizarLimiar(self.w0[atualizacoes], self.n, (self.d[i] - y)))
                    atualizacoes += 1

                epoca += 1
                erro = True

    def calculoSaida(self, x, w, w0): #funcao ira calcular o U

        resultado = 0

        for i in range(len(x)):
            resultado += x[i]*w[i]
        resultado -= w0

        return resultado

    def atualizarPesos(self, w, n, erro, x): #funcao que irá atualizar os pesos
        resultado = []

        for i in range(len(w)):
            resultado.append(
                w[i] + n*erro*x[i]
            )

        return resultado

    def autalizarLimiar(self, w0, n, erro):
        return w0 + n*erro*(-1)


    def resultado(self):
        print(self.w)

