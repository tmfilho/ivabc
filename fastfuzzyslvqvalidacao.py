from numpy import *

import old
import old.fastslvq as slvq
import old.fcm as fcm

from util.functions import print_confusion_matrix


def treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, taxa = 0.3):
    k = shape(prototipos)[0]
    n = shape(treinamento)[0]
    taxas = ones(k) * taxa
    passos = 0
    parar = 0
    quantosPararam = 0
    pesosPrototipos = prod(pesos,1)
    p = shape(prototipos)[1]/2
    deltas = zeros((shape(prototipos)[0], p))
    
    distancias = old.fcm.calcularDistancias(prototipos, treinamento, pesos, n, k)
    graus = old.fcm.calcularGraus(distancias)
    Jatual = old.fcm.calcularCriterio(graus, distancias, classesTreinamento, classesPrototipos)
    
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
            #dists = (sum(mins + maxs,1) + 0.0000000001)**-2.0
            
            #graus = dists / sum(dists)
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
            pesos = old.fcm.calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, n, k)
            distancias = old.fcm.calcularDistancias(prototipos, treinamento, pesos, n, k)
            graus = old.fcm.calcularGraus(distancias)
            Jdepois = old.fcm.calcularCriterio(graus, distancias, treinamento, classesTreinamento, classesPrototipos)
            if Jdepois < Jatual:
                parar = 0
                configuracaoAtual = copy(prototipos)
                pesosAtuais = copy(pesos)
            else:
                parar = parar + 1
            Jatual = Jdepois
    prototipos = copy(configuracaoAtual)   
    pesos = copy(pesosAtuais)   
    return [prototipos, pesos]

def testar(teste, classesTeste, particula, classesParticula, pesos):
    k = shape(particula)[0]
    n = shape(teste)[0]
    distancias = old.fcm.calcularDistancias(particula, teste, pesos, n, k)
    graus = old.fcm.calcularGraus(distancias)
    particao = argmax(graus,1)       
    classesResultantes = array([classesParticula[prot] for prot in particao])
    numeroErros = float(size(classesTeste[classesTeste != classesResultantes]))
    # return (numeroErros / n)*100.0
    return classesResultantes

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
            [treinamento, classesTreinamento, teste, classesTeste] = slvq.separarConjuntos(folds, dadosEmbaralhados, classesEmbaralhadas, fold)
            preds = zeros((30, len(classesTeste)))
            from tqdm import tqdm
            for l in tqdm(arange(30)):
                [prototipos, classesPrototipos] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nMedias)
                [prototipos, pesos, _] = old.fcm.treinar(treinamento,
                                                       classesTreinamento, prototipos, classesPrototipos, 500)
                # print testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
                [prototipos, pesos] = treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, 0.3)
                preds[l] = testar(teste, classesTeste, prototipos,
                           classesPrototipos, pesos)
            predictions = around(mean(preds, axis=0))
            print_confusion_matrix(classesTeste, predictions)
            exit()
            # erros[i*nFolds + fold] = testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
            # print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)
    
if __name__ == "__main__":
    random.seed(1)

    [dados, classes] = slvq.lerDados(
        "base_meses_corrigida_nao_normalizados.txt")
    #rodarHoldOut(dados, classes, 1, [1,1], 100)
    rodarValidacaoCruzada(dados, classes, [34, 28, 12, 20, 12, 42, 38, 34, 6, 40], 10, 10)