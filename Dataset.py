class Dataset:

    def __init__(self, arquivoD, arquivoX):
        self.arquivoD = arquivoD
        self.arquivoX = arquivoX

    def definindoValoresDesejados(self): #função que irá retornar os valores desejados (d) já convertidos para INT

        dados = []
        d = []

        dados = self.arquivoD.readlines()

        for i in dados:
            d.append(int(i))

        return d

    def definindoEntradas(self): #função que irá retornar conjuntos de vetores com as entradas (x) convertidos em FLOAT

        dados = []
        linha = []
        x = []

        dados = self.arquivoX.readlines()

        for i in dados:
            j = i.split()
            for k in j:
                linha.append(float(k))
            x.append(linha)
            linha = []

        return x








