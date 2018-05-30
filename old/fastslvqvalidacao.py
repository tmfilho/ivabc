from numpy import *
import batchlvq as blvq
import fastslvq as slvq

def separarConjuntos(folds, dados, classes, parteTeste): 
    parteValidacao = parteTeste + 1
    if parteValidacao == len(folds):
        parteValidacao = 0 
    treinamento = array([dados[i,:] for fold in delete(arange(len(folds)),[parteTeste, parteValidacao]) for i in folds[fold]])
    classesTreinamento = array([classes[i] for fold in delete(arange(len(folds)),[parteTeste, parteValidacao]) for i in folds[fold]])
    teste = dados[folds[parteTeste],:]
    classesTeste = classes[folds[parteTeste]]
    validacao = dados[folds[parteValidacao],:]
    classesValidacao = classes[folds[parteValidacao]]
    return [treinamento, classesTreinamento, teste, classesTeste, validacao, classesValidacao]

def treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, validacao, classesValidacao, pesos, taxa = 0.3):
    k = shape(prototipos)[0]
    n = shape(treinamento)[0]
    taxas = ones(k) * taxa
    passos = 0
    parar = 0
    quantosPararam = 0
    erroValidacao = 101
    configuracaoAtual = copy(prototipos)
    pesosAtuais = copy(pesos)
    while parar < 3 and passos < 500:
        ciclos = 0
        t = len(treinamento) * len(prototipos)
        while ciclos < t:
            i = random.choice(n)
            ds = treinamento[i,:]
            
            mins = pesos*((prototipos[:,::2] - ds[::2])**2)
            maxs = pesos*((prototipos[:,1::2] - ds[1::2])**2)
            dists = sum(mins + maxs,1)
            
            indice = argmin(dists)
            acertou = -1.0
            if classesTreinamento[i] == classesPrototipos[indice]:
                acertou = 1.0
            prototipos[indice,:] = prototipos[indice,:] + acertou * taxas[indice] * (ds - prototipos[indice,:])
            taxas[indice] = taxas[indice]/(1.0 +(acertou*taxas[indice]))
            if ((taxas[indice] != 0) and ((taxas[indice] < 0.0001) or (taxas[indice] >= 1.0))):
                taxas[indice] = 0
                quantosPararam += 1
            if quantosPararam >= k:
                parar = 3
                break
            ciclos += 1
        passos += 1 
        if parar != 3:
            pesos = blvq.calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, k, n)
            erroNovo = slvq.testar(validacao, classesValidacao, prototipos, classesPrototipos, pesos)
            if erroNovo < erroValidacao:
                parar = 0
                erroValidacao = erroNovo
                configuracaoAtual = copy(prototipos)
                pesosAtuais = copy(pesos)
            else:
                parar = parar + 1
                erroValidacao = erroNovo
    prototipos = copy(configuracaoAtual)   
    pesos = copy(pesosAtuais)   
    return [prototipos, pesos]

def rodarValidacaoCruzada(dados, classes, nMedias, montecarlo, nFolds):  
    erros = zeros(montecarlo*nFolds)
    n = size(classes)
    for i in range(montecarlo):
        indices = arange(n)
        random.shuffle(indices)
        dadosEmbaralhados = dados[indices,:]
        classesEmbaralhadas = classes[indices]
        folds = slvq.separarFolds(dadosEmbaralhados, classesEmbaralhadas, nFolds)
        for fold in range(nFolds):
            print i*nFolds + fold
            [treinamento, classesTreinamento, teste, classesTeste, validacao, classesValidacao] = separarConjuntos(folds, dadosEmbaralhados, classesEmbaralhadas, fold)      
            [prototipos, classesPrototipos] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nMedias)
            [prototipos, pesos] = blvq.treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, 500)
            [prototipos, pesos] = treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, validacao, classesValidacao, pesos)
            erros[i*nFolds + fold] =  slvq.testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
    print erros
    print mean(erros)
    print std(erros)
    
if __name__ == "__main__":
    random.seed(1)
    
    [dados, classes] = slvq.lerDados("mediterraneo_oceanico_normalizados.txt")
    #rodarHoldOut(dados, classes, 1, [1,1], 100)
    rodarValidacaoCruzada(dados, classes, [10,22], 10, 10)