from numpy import *

def inicializarGraus(n,k):
    graus = random.rand(n,k)
    return graus/sum(graus,1,keepdims=True)       

def getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot, indices = False):
    if indices:
        return array([i for i in where(particao == prot)[0] if classesTreinamento[i] == classesPrototipos[prot]])
    else:
        return array([treinamento[i,:] for i in where(particao == prot)[0] if classesTreinamento[i] == classesPrototipos[prot]])

def calcularPrototipos(graus, treinamento, classesTreinamento, prototipos, classesPrototipos, particao, k):
    novosPrototipos = zeros(shape(prototipos))
    for prot in range(k):
        membros = getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot, True)
        if size(membros) > 0:
            grausMembros = graus[membros,prot] ** 2.0
            xMembros = treinamento[membros,:]
            novosPrototipos[prot,:] = sum((xMembros.T*grausMembros).T,0) / sum(grausMembros)
        else:
            novosPrototipos[prot,:] = prototipos[prot,:]
    return novosPrototipos

def calcularGraus(distancias):
    d = distancias + + 0.0000000001
    return (d*sum(1.0/d,1,keepdims=True))**-1

def calcularCriterio(graus, distancias, classesTreinamento, classesPrototipos):
    particao = argmax(graus,1)   
    k = len(classesPrototipos)
    for prot in range(k):
        errados = [i for i in where(particao==prot)[0] if classesTreinamento[i] != classesPrototipos[prot]]
        graus[errados,:] = 0
        graus[where(particao!=prot)[0],prot] = 0
    return sum((graus**2)*distancias)

def calcularDistancias(prototipos, treinamento, pesos, n, k):
    distancias = zeros((n,k))
    for prot in range(k):
        mins = pesos[prot,:]*((prototipos[prot,::2] - treinamento[:,::2])**2)
        maxs = pesos[prot,:]*((prototipos[prot,1::2] - treinamento[:,1::2])**2)
        distancias[:,prot] = sum(mins + maxs,1) + 0.0000000001
    return distancias

def treinar(treinamento, classesTreinamento, prototipos, classesPrototipos, tMax):
    t = 1
    pesos = ones ((shape(prototipos)[0], shape(prototipos)[1]/2))
    deltas = zeros ((shape(prototipos)[0], shape(prototipos)[1]/2))
    k = shape(prototipos)[0]
    n = shape(treinamento)[0]
    epsilon = 0.00001
    Jatual = 1
    Jdepois = -1
    while t <= tMax and abs(Jatual - Jdepois)>epsilon:
        #fase dos prototipos
        distancias = calcularDistancias(prototipos, treinamento, pesos, n, k)
        graus = calcularGraus(distancias)  
        
        Jatual = Jdepois
        Jdepois = calcularCriterio(graus, distancias, classesTreinamento, classesPrototipos)
                  
        particao = argmax(graus,1)        
        prototipos = calcularPrototipos(graus, treinamento, classesTreinamento, prototipos, classesPrototipos, particao, k)
                
        t = t + 1
        #fase dos pesos
        [pesos, deltas] = calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, n, k)
    return [prototipos, pesos, deltas]    

def calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, pesos, n, k):
    nClasses = max(classesTreinamento + 1)
    distancias = calcularDistancias(prototipos, treinamento, pesos, n, k)
    graus = calcularGraus(distancias)  
    
    particao = argmax(graus,1)
    distancias = (graus**2)*distancias
    deltas = zeros((shape(prototipos)[0], shape(prototipos)[1]/2))
    for prot in range(k):
        membros = getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot)
        if size(membros) > 0:
            mins = (prototipos[prot,::2] - membros[:,::2])**2
            maxs = (prototipos[prot,1::2] - membros[:,1::2])**2
            grausMembros = (graus[getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot,True),prot]**2)
            deltas[prot,:] = sum(grausMembros.reshape((size(grausMembros),1))*(mins + maxs),0)
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
    return [array([v/deltas[i,:] if all(deltas[i,:] <> 0) else ones(p) * pesosPrototipos[i]**(1.0/p) for i,v in enumerate(prods)]),deltas]