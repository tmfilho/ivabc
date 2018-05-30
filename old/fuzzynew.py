from numpy import *

import fastslvq as slvq
import fcm as fcm


def separarConjuntos(folds, dados, classes, parteTeste):
    parteValidacao = parteTeste + 1
    if parteTeste == len(folds) - 1:
        parteValidacao = 0
    treinamento = array([dados[i,:] for fold in delete(arange(len(folds)), [parteTeste, parteValidacao]) for i in folds[fold]])
    classesTreinamento = array([classes[i] for fold in delete(arange(len(folds)), [parteTeste, parteValidacao]) for i in folds[fold]])
    teste = dados[folds[parteTeste],:]
    classesTeste = classes[folds[parteTeste]]
    validacao = dados[folds[parteValidacao],:]
    classesValidacao = classes[folds[parteValidacao]]
    return [treinamento, classesTreinamento, teste, classesTeste, validacao, classesValidacao]

def treinar(treinamento, classesTreinamento, validacao, classesValidacao, prototipos, classesPrototipos, pesos, somatorios, taxa = 0.3):
    k = shape(prototipos)[0]
    n = shape(treinamento)[0]
    taxas = ones(k) * taxa
    passos = 0
    parar = 0
    quantosPararam = 0
    pesosPrototipos = prod(pesos,1)
    p = shape(prototipos)[1]/2
    deltas = somatorios
    
    erroAtual = testar(validacao, classesValidacao, prototipos, classesPrototipos, pesos)
    
    print erroAtual
    configuracaoAtual = copy(prototipos)
    pesosAtuais = copy(pesos)
    
    while parar < 3 and passos < 500:
        ciclos = 0
        while ciclos < len(treinamento) * len(prototipos):
            i = random.choice(n)
            ds = treinamento[i,:]
            
            mins = pesos*((prototipos[:,::2] - ds[::2])**2)
            maxs = pesos*((prototipos[:,1::2] - ds[1::2])**2)
            dists = sum(mins + maxs,1) + 0.0000000001
            graus = (dists*sum(1.0/dists,keepdims=True))**-1
            
            for indice in range (k):
                acertou = -1.0
                if classesTreinamento[i] == classesPrototipos[indice]:
                    acertou = 1.0
                    mins = (prototipos[indice,::2] - ds[::2])**2
                    maxs = (prototipos[indice,1::2] - ds[1::2])**2
                    temp = graus[indice]**2 * (mins + maxs)
                    deltas[indice,:] = (1.0-taxas[indice]) * deltas[indice,:] + taxas[indice] * temp
                    produtorio = (pesosPrototipos[indice]*prod(deltas[indice,:])) ** (1.0/p)
                    if all(deltas[indice,:] <> 0):            
                        pesos[indice,:] = produtorio / deltas[indice,:]
                prototipos[indice,:] = prototipos[indice,:] + acertou * graus[indice] * taxas[indice] * (ds - prototipos[indice,:])
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
            [pesos, deltas] = fcm.calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, n, k)
            novoErro = testar(validacao, classesValidacao, prototipos, classesPrototipos, pesos)
            
            print novoErro
            if novoErro < erroAtual:
                parar = 0
                configuracaoAtual = copy(prototipos)
                pesosAtuais = copy(pesos)
            else:
                parar = parar + 1
            erroAtual = novoErro
    prototipos = copy(configuracaoAtual)   
    pesos = copy(pesosAtuais)   
    return [prototipos, pesos]

def testar(teste, classesTeste, particula, classesParticula, pesos):
    k = shape(particula)[0]
    n = shape(teste)[0]
    distancias = fcm.calcularDistancias(particula, teste, pesos, n, k)
    graus = fcm.calcularGraus(distancias)  
    particao = argmax(graus,1)       
    classesResultantes = array([classesParticula[prot] for prot in particao])
    numeroErros = float(size(classesTeste[classesTeste != classesResultantes]))
    return (numeroErros / n)*100.0

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
            [prototipos, pesos, deltas] = fcm.treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, 500)
            print testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
            [prototipos, pesos] = treinar(treinamento, classesTreinamento, validacao, classesValidacao, prototipos, classesPrototipos, pesos, deltas, 0.3)
            erros[i*nFolds + fold] =  testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
            print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)
    
if __name__ == "__main__":
    random.seed(1)
    
    [dados, classes] = slvq.lerDados("mediterraneo_oceanico_normalizados.txt")
    #rodarHoldOut(dados, classes, 1, [1,1], 100)
    rodarValidacaoCruzada(dados, classes, [10,22], 10, 10)