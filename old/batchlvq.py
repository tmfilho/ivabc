from numpy import *

def getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot):
    return array([treinamento[i,:] for i in where(particao == prot)[0] if classesTreinamento[i] == classesPrototipos[prot]])

def treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, tMax):
    t = 1
    continuar = True
    pesos = ones ((shape(prototipos)[0], shape(prototipos)[1]/2))
    while t < tMax and continuar:
        k = shape(prototipos)[0]
        n = shape(treinamento)[0]
        #fase dos prototipos
        distancias = zeros((n,k))
        for prot in range(k):
            mins = pesos[prot,:]*((prototipos[prot,::2] - treinamento[:,::2])**2)
            maxs = pesos[prot,:]*((prototipos[prot,1::2] - treinamento[:,1::2])**2)
            distancias[:,prot] = sum(mins + maxs,1)
        particao = argmin(distancias,1)
                
        novosProts = zeros(shape(prototipos))
        for prot in range(k):
            membros = getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot)
            if size(membros) > 0:
                novosProts[prot,:] = mean(membros,0)
            else:
                novosProts[prot,:] = prototipos[prot,:]
        if all(novosProts == prototipos):
            continuar = False
        else:
            t = t + 1
        prototipos = novosProts
        #fase dos pesos
        pesos = calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, k, n)
    return [prototipos, pesos]    

def calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, k, n):
    nClasses = max(classesTreinamento + 1)
    distancias = zeros((n,k))
    for prot in range(k):
        mins = pesos[prot,:]*((prototipos[prot,::2] - treinamento[:,::2])**2)
        maxs = pesos[prot,:]*((prototipos[prot,1::2] - treinamento[:,1::2])**2)
        distancias[:,prot] = sum(mins + maxs,1)
    particao = argmin(distancias,1)
    deltas = zeros((shape(prototipos)[0], shape(prototipos)[1]/2))
    for prot in range(k):
        membros = getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot)
        if size(membros) > 0:
            mins = (prototipos[prot,::2] - membros[:,::2])**2
            maxs = (prototipos[prot,1::2] - membros[:,1::2])**2
            deltas[prot,:] = sum(mins + maxs,0)
        errados = [i for i in where(particao==prot)[0] if classesTreinamento[i] != classesPrototipos[prot]]
        distancias[errados,:] = 0
        distancias[where(particao!=prot)[0],prot] = 0
    somatoriosClasses = array([])
    for classe in range(nClasses):
        somatoriosClasses = append(somatoriosClasses, sum(distancias[:,where(classesPrototipos == classe)[0]]))
    if any(somatoriosClasses == 0):
        pesosClasses = ones(nClasses)
    else:
        pesosClasses = (prod(somatoriosClasses)**(1.0/nClasses))/somatoriosClasses
    pesosPrototipos = ones(k)
    for classe in range(nClasses):
        somatorioPrototipos = sum(distancias[:,where(classesPrototipos == classe)[0]],axis=0)
        protsClasse = where(classesPrototipos == classe)[0]
        if any(somatorioPrototipos == 0):
            pesosPrototipos[protsClasse] = ones(size(protsClasse)) * pesosClasses[classe]**(1.0/size(protsClasse))
        else:
            pesosPrototipos[protsClasse] = ((pesosClasses[classe]*prod(somatorioPrototipos))**(1.0/size(protsClasse)))/somatorioPrototipos
    p = shape(prototipos)[1]/2
    prods = (pesosPrototipos*prod(deltas,1))**(1.0/p)
    return array([v/deltas[i,:] if all(deltas[i,:] <> 0) else ones(p) * pesosPrototipos[i]**(1.0/p) for i,v in enumerate(prods)])