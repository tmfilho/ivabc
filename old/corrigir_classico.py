from numpy import *

def lerDados(nome):
    pasta = "dadosC/"
    with open(pasta + nome, 'r') as f:
        dados = array([line.split() for line in f])
        dados = dados.astype(float)
        #dados = dados[dados[:,shape(dados)[1]-1].argsort()]
        classes = dados[:,shape(dados)[1]-1].astype(int)
        dados = dados[:,0:shape(dados)[1]-1]
    #dados = dados[(classes == 3).__or__(classes==5)]
    #temp = classes[(classes == 3).__or__(classes==5)]
    #classes = temp
    #for c in range(len(temp)):
    #    if temp[c] == 3:
    #        classes[c] = 0
    #    elif temp[c] == 5:
    #        classes[c] = 1
            
    return [dados, classes]

def ajustarMinsMaxs(dados):
    minimos = dados[:,::2]
    maximos = dados[:,1::2]
    ind = minimos > maximos
    if any(ind == True):
        temp = minimos[ind]
        minimos[ind] = maximos[ind]
        maximos[ind] = temp
        
        dados[:,::2] = minimos
        dados[:,1::2] = maximos   
    return dados

if __name__ == "__main__":
    nome = "dadosSegSemConstantes.csv"
    [dados, classes] = lerDados(nome)
    pa = shape(dados)[1]
    for var in arange(pa):
        dados[:,var] = (dados[:,var] - dados[:,var].min())/(dados[:,var].max() - dados[:,var].min())
    
    n = shape(dados)[0]    
    with open("dadosC/dadosSegSemConstantes_corrigida.txt", 'a') as f:
        for dado in arange(n):
            linha = ""
            for var in arange(pa):
                linha = linha + "{0} ".format(dados[dado,var])
            
            linha = linha + "{0}\n".format(classes[dado])    
            f.write(linha)

    