# To change this template, choose Tools | Templates
# and open the template in the editor.

from numpy import *
import batchlvq as blvq

def lerDados(nome):
    with open("dados/"+nome, 'r') as f:
        dados = array([line.split() for line in f])
        dados = dados.astype(float)
        dados = dados[dados[:,shape(dados)[1]-1].argsort()]
        classes = dados[:,shape(dados)[1]-1].astype(int)
        dados = dados[:,0:shape(dados)[1]-1]
    return [dados, classes]

def separarHoldOut(dados, classes, percentagem):
    if percentagem < 1:
        nClasses = max(classes + 1)
        for classe in range(nClasses):
            membros = where(classes == classe)[0]
            nTreinamento = floor(size(membros) * percentagem)
            nTeste = size(membros) - nTreinamento
            membrosTreinamento = random.choice(membros, nTreinamento, replace=False)
            membrosTeste = delete(membros,membrosTreinamento)
            if classe > 0:
                treinamento = append(treinamento, dados[membrosTreinamento,:], axis = 0)
                classesTreinamento = append(classesTreinamento, classes[membrosTreinamento])
                teste = append(teste, dados[membrosTeste,:], axis = 0)
                classesTeste = append(classesTeste, classes[membrosTeste])
            else:
                treinamento = dados[membrosTreinamento,:]
                classesTreinamento = classes[membrosTreinamento]
                teste = dados[membrosTeste,:]
                classesTeste = classes[membrosTeste]
    else:
        n = shape(dados)[0]
        escolhido = random.choice(n)
        naoEscolhidos = arange(n-1)
        treinamento = dados[naoEscolhidos,:]
        classesTreinamento = classes[naoEscolhidos]
        teste = array([dados[escolhido,:]])
        classesTeste = array([classes[escolhido]])
            
    return [treinamento, classesTreinamento, teste, classesTeste]

def separarFolds(dados, classes, nFolds):
    nClasses = max(classes + 1)
    folds = range(nFolds)
    for classe in range(nClasses):
        membros = where(classes == classe)[0]
        nMembros = int(floor(size(membros) / nFolds))
        inicio = 0
        fim = nMembros
        for fold in range(nFolds):
            if classe == 0:
                folds[fold] = membros[inicio:fim]
            else:
                folds[fold] = append(folds[fold], membros[inicio:fim])
            if fold < nFolds - 1:    
                inicio = fim
                fim = fim + nMembros
        fold = 0
        for sobra in range(int(fim),size(membros)):
            folds[fold] = append(folds[fold], membros[sobra])
            fold = fold + 1
    return folds

def separarConjuntos(folds, dados, classes, parteTeste): 
    treinamento = array([dados[i,:] for fold in delete(arange(len(folds)),parteTeste) for i in folds[fold]])
    classesTreinamento = array([classes[i] for fold in delete(arange(len(folds)),parteTeste) for i in folds[fold]])
    teste = dados[folds[parteTeste],:]
    classesTeste = classes[folds[parteTeste]]
    return [treinamento, classesTreinamento, teste, classesTeste]

def iniciarPrototiposPelaMedia(treinamento, classesTreinamento):
    nClasses = max(classesTreinamento + 1)
    prototipos = zeros((nClasses,shape(treinamento)[1]))
    classesPrototipos = arange(nClasses)
    for classe in range(nClasses):
        prototipos[classe,:] = mean(treinamento[where(classesTreinamento == classe)[0],:],0)
    return [prototipos, classesPrototipos] 

def iniciarPrototiposPorNMedias(treinamento, classesTreinamento, nMedias):
    nClasses = max(classesTreinamento + 1)
    prototipos = zeros((sum(nMedias),shape(treinamento)[1]))
    classesPrototipos = zeros(sum(nMedias))
    prot = 0
    for classe in range(nClasses):
        membros = where(classesTreinamento == classe)[0]
        nPorPrototipo = floor(size(membros)/nMedias[classe])
        random.shuffle(membros)
        inicio = 0
        fim = nPorPrototipo
        for i in range(nMedias[classe]):
            if i == nMedias[classe] - 1:
                fim = size(membros)
            prototipos[prot,:] = mean(treinamento[membros[inicio:fim],:],0)
            inicio = fim
            fim = fim + nPorPrototipo
            classesPrototipos[prot] = classe
            prot = prot + 1
    return [prototipos, classesPrototipos] 

def iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts):
    nClasses = max(classesTreinamento + 1)
    prototipos = zeros((sum(nProts),shape(treinamento)[1]))
    classesPrototipos = zeros(sum(nProts))
    inicio = 0
    fim = 0
    for classe in range(nClasses):
        inicio = fim
        fim = fim + nProts[classe]
        membros = where(classesTreinamento == classe)[0]
        escolhidos = random.choice(membros, nProts[classe])
        prototipos[inicio:fim,:] = copy(treinamento[escolhidos,:])
        classesPrototipos[inicio:fim] = classesTreinamento[escolhidos]
    return [prototipos, classesPrototipos] 

def iniciarPrototiposPorSelecaoPorClasse(treinamento, classesTreinamento, nProts, classe):
    membros = where(classesTreinamento == classe)[0]
    escolhidos = random.choice(membros, nProts)
    prototipos = copy(treinamento[escolhidos,:])
    classesPrototipos = classesTreinamento[escolhidos]
    return [prototipos, classesPrototipos] 

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
    while parar < 3 and passos < 10000:
        i = random.choice(n)
        ds = treinamento[i,:]
        
        mins = pesos*((prototipos[:,::2] - ds[::2])**2)
        maxs = pesos*((prototipos[:,1::2] - ds[1::2])**2)
        dists = sum(mins + maxs,1)
        
        indice = argmin(dists)
        acertou = -1.0
        if classesTreinamento[i] == classesPrototipos[indice]:
            acertou = 1.0
            #mins = (prototipos[indice,::2] - ds[::2])**2
            #maxs = (prototipos[indice,1::2] - ds[1::2])**2
            #temp = mins + maxs
            #deltas[indice,:] = (1.0-taxas[indice]) * deltas[indice,:] + taxas[indice] * temp
            #produtorio = (pesosPrototipos[indice]*prod(deltas[indice,:])) ** (1.0/p)
            #if all(deltas[indice,:] <> 0):            
            #    pesos[indice,:] = produtorio / deltas[indice,:]
        prototipos[indice,:] = prototipos[indice,:] + acertou * taxas[indice] * (ds - prototipos[indice,:])
        taxas[indice] = taxas[indice]/(1.0 +(acertou*taxas[indice]))
        if ((taxas[indice] != 0) and ((taxas[indice] < 0.0001) or (taxas[indice] >= 1.0))):
            taxas[indice] = 0
            quantosPararam += 1
        if quantosPararam >= k:
            parar = 3
            break
        passos += 1   
    return [prototipos, pesos]

def testar(teste, classesTeste, particula, classesParticula, pesos):
    k = shape(particula)[0]
    n = shape(teste)[0]
    distancias = zeros((n,k))
    for prot in range(k):
        mins = pesos[prot,:]*((particula[prot,::2] - teste[:,::2])**2)
        maxs = pesos[prot,:]*((particula[prot,1::2] - teste[:,1::2])**2)
        distancias[:,prot] = sum(mins + maxs,1)
    particao = argmin(distancias,1)
    classesResultantes = array([classesParticula[prot] for prot in particao])
    numeroErros = float(size(classesTeste[classesTeste != classesResultantes]))
    return (numeroErros / n)*100.0

def rodarHoldOut(dados, classes, percentagem, nMedias, montecarlo):
    erros = zeros(montecarlo)
    for i in range(montecarlo):
        print i
        [treinamento, classesTreinamento, teste, classesTeste] = separarHoldOut(dados, classes, percentagem)        
        [prototipos, classesPrototipos] = iniciarPrototiposPorNMedias(treinamento, classesTreinamento, nMedias)
        [prototipos, pesos] = blvq.treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, 500)
        [prototipos, pesos] = treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos)
        erros[i] =  testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
    print erros
    print mean(erros)
    print std(erros)
 
def rodarValidacaoCruzada(dados, classes, nMedias, montecarlo, nFolds):  
    erros = zeros(montecarlo*nFolds)
    n = size(classes)
    for i in range(montecarlo):
        indices = arange(n)
        random.shuffle(indices)
        dadosEmbaralhados = dados[indices,:]
        classesEmbaralhadas = classes[indices]
        folds = separarFolds(dadosEmbaralhados, classesEmbaralhadas, nFolds)
        for fold in range(nFolds):
            print i*nFolds + fold
            [treinamento, classesTreinamento, teste, classesTeste] = separarConjuntos(folds, dadosEmbaralhados, classesEmbaralhadas, fold)      
            [prototipos, classesPrototipos] = iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nMedias)
            [prototipos, pesos] = blvq.treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, 500)
            [prototipos, pesos] = treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos)
            erros[i*nFolds + fold] =  testar(teste, classesTeste, prototipos, classesPrototipos, pesos)
    print erros
    print mean(erros)
    print std(erros)

if __name__ == "__main__":
    random.seed(1)
    
    [dados, classes] = lerDados("cogumelos2Classes.txt")
    #rodarHoldOut(dados, classes, 1, [1,1], 100)
    rodarValidacaoCruzada(dados, classes, [7,2], 10, 10)