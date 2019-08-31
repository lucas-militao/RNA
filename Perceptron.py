class Perceptron:

    def __init__(self, entradas, saidasDesejadas, pesosSinapticos, taxaAprendizagem, limiar):
        self.x = entradas
        self.d = saidasDesejadas
        self.w = pesosSinapticos
        self.n = taxaAprendizagem
        self.w0 = limiar
        self.u = []

    def treinamento(self): #funcao que ira realizar o treinamento

        self.epoca = 0
        self.y = 0
        self.erro = False

        while(self.erro != False): #estou um pouco enrolado nesse loop

            self.u.append(self.calculoSaida(self.x[self.epoca], self.w[self.epoca], self.w0[self.epoca]))

            y = 1 if self.u[self.epoca] >= 0 else -1

            if(y != self.d[self.epoca]):
                for i in self.w[self.epoca]:
                    self.w.append(
                        self.atualizarPesos(self.w, self.n, (self.d[self.epoca] - self.y), self.x[self.epoca])
                    )

                    self.w0.append(
                        self.w0 + self.n*(self.d[self.epoca] - self.y[self.epoca]) * (-1)
                    )

                self.erro = True
                self.treinamento()

            else:
                self.epoca += 1

    def calculoSaida(self, x, w, w0): #funcao ira calcular o U

        self.resultado = 0

        for i in x:
            self.resultado += x[i]*w[i]
        self.resultado -= w0

        return self.resultado

    def atualizarPesos(self, w, n, erro, x): #funcao que irá atualizar os pesos

        self.resultado = []

        for i in w:
            self.resultado.append(
                w[i] + n*erro*x[i]
            )

        return self.resultado


