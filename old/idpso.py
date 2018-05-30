from numpy import *

import fastslvq as slvq
import fcm as fcm


def inicializar(treinamento, classesTreinamento, nProts, np, intervalar = True):
    if intervalar:
        p = shape(treinamento)[1]/2
        mins = treinamento[:,::2].min(0)
        maxs = treinamento[:,1::2].max(0)
    else:
        mins = treinamento.min(0)
        maxs = treinamento.max(0)
        p = shape(treinamento)[1]
    particulas = zeros((np,sum(nProts)+1,shape(treinamento)[1]))
    velocidades = random.rand(np,sum(nProts)+1,shape(treinamento)[1])
    classesParticulas = zeros((np,sum(nProts)))
    criterios = zeros(np)
    for particula in range(np):
        variaveisConsideradas = random.rand(shape(treinamento)[1])
        [prototipos, classesPrototipos] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts)        
        variaveisConsideradas[p:] = 0
        particulas[particula] = append([variaveisConsideradas],prototipos,axis=0)
        classesParticulas[particula] = classesPrototipos
        particulas[particula] = ajustarMinsMaxs(ajustarVariaveisConsideradas(particulas[particula],p), mins, maxs, intervalar)
        pesos = calcularPesosPrototipos(treinamento, classesTreinamento, particulas[particula,1:], classesParticulas[particula], particulas[particula,0,:p], shape(treinamento)[0], shape(particulas[particula])[0]-1, p, intervalar)
        criterios[particula] = calcularCriterioJ(pesos, particulas[particula], classesParticulas[particula], treinamento, classesTreinamento, p, intervalar)
    PBEST = copy(particulas)
    criteriosPBEST = copy(criterios)
    indice = argmin(criterios)
    GBEST = copy(particulas[indice])
    classesGBEST = copy(classesParticulas[indice])
    criterioGBEST = criterios[indice]
    return [particulas, velocidades, classesParticulas, PBEST, criteriosPBEST, GBEST, classesGBEST, criterioGBEST, mins, maxs]

def calcularCriterioJ(pesos, particula, classesParticula, treinamento, classesTreinamento, p, intervalar = True):
    erro = testar(treinamento, classesTreinamento, particula, classesParticula, pesos, intervalar)
    return erro / (sum (particula[0,:p]) + 0.0000000001)
 
def calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k, intervalar = True):
    consideradas = around(variaveisConsideradas)
    distancias = zeros((n,k))
    for prot in range(k):
        if intervalar:
            mins = consideradas * pesos[prot,:] * ((prototipos[prot,::2] - treinamento[:,::2])**2)
            maxs = consideradas * pesos[prot,:] * ((prototipos[prot,1::2] - treinamento[:,1::2])**2)
            distancias[:,prot] = sum(mins + maxs,1) + 0.0000000001
        else:
            distancias[:,prot] = sum((consideradas * pesos[prot,:] * (prototipos[prot,:] - treinamento)**2),1) + 0.0000000001
    return distancias

def calcularFi(particula, PBEST, GBEST):
    fi = -1
    numerador = sqrt(sum((GBEST-particula)**2.0))
    denominador = sqrt(sum((PBEST-particula)**2.0))
    if denominador != 0:
        fi = numerador/denominador
    return fi

def atualizarMelhores(particula, classesParticula, PBEST, criterioPBEST, GBEST, criterioGBEST, classesGBEST, treinamento, classesTreinamento, p, intervalar = True):
    pesos = calcularPesosPrototipos(treinamento, classesTreinamento, particula[1:], classesParticula, particula[0,:p], shape(treinamento)[0], shape(particula)[0]-1, p, intervalar)
    criterio = calcularCriterioJ(pesos, particula, classesParticula, treinamento, classesTreinamento, p, intervalar)
    if criterio < criterioPBEST:
        criterioPBEST = criterio
        PBEST = copy(particula)
    if criterio < criterioGBEST:
        criterioGBEST = criterio
        GBEST = copy(particula)
        classesGBEST = copy(classesParticula)   
    return [PBEST, criterioPBEST, GBEST, criterioGBEST, classesGBEST]

def ajustarVariaveisConsideradas(particula,p):
    consideradas = particula[0,:p]
    menorQue0 = where(consideradas < 0)[0]
    consideradas[menorQue0] = 0
    maiorQue1 = where(consideradas > 1)[0]
    consideradas[maiorQue1] = 1
    while all(around(consideradas) == 0):
        consideradas = random.rand(len(consideradas))
    particula[0,:p] = consideradas
    return particula

def ajustarMinsMaxs(dados, mins, maxs, intervalar = True):
    prototipos = dados[1:]
    k = shape(prototipos)[0]
    if intervalar:
        p = shape(prototipos)[1]/2
        for prot in range(k):
            minsProt = prototipos[prot,::2]
            maxsProt = prototipos[prot,1::2]
            for var in range(p):
                if minsProt[var] < mins[var]:
                    minsProt[var] = mins[var]
                if minsProt[var] > maxs[var]:
                    minsProt[var] = maxs[var]
                if maxsProt[var] > maxs[var]:
                    maxsProt[var] = maxs[var]
                if maxsProt[var] < mins[var]:
                    maxsProt[var] = mins[var]
                if minsProt[var] > maxsProt[var]:
                    temp = maxsProt[var]
                    maxsProt[var] = minsProt[var]
                    minsProt[var] = temp
            prototipos[prot:,::2] = minsProt
            prototipos[prot:1,::2] = maxsProt
    else:
        p = shape(prototipos)[1]
        for prot in range(k):
            prototipo = prototipos[prot,:]
            for var in range(p):
                if prototipo[var] < mins[var]:
                    prototipo[var] = mins[var]
                if prototipo[var] > maxs[var]:
                    prototipo[var] = maxs[var]
            prototipos[prot,:] = prototipo
    dados[1:] = prototipos  
    return dados

def calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesPrototipos, variaveisConsideradas, n, k, p, intervalar = True, pesos = []):
    nClasses = max(classesTreinamento + 1)
    consideradas = around(variaveisConsideradas)
    if len(pesos) == 0:
        pesos = ones((k,p))
    distancias = calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k, intervalar)
    graus = fcm.calcularGraus(distancias)  
    
    particao = argmax(graus,1)
    distancias = (graus**2)*distancias
    
    pesos = ones((k, p))
    for prot in range(k):
        membros = fcm.getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot)
        if size(membros) > 0:
            if intervalar:
                mins = (prototipos[prot,::2] - membros[:,::2])**2
                maxs = (prototipos[prot,1::2] - membros[:,1::2])**2
                diff = mins + maxs
            else:
                diff = (prototipos[prot] - membros)**2
            grausMembros = (graus[
                                fcm.getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot, True), prot] ** 2)
            deltas = sum(grausMembros.reshape((size(grausMembros),1)) * consideradas * (diff + 0.0000000001),0)
            produtorio = prod(deltas[deltas > 0])**(1.0/sum(consideradas))
            pesos[prot,:] = array([produtorio / v if v > 0 else 0 for v in deltas])
            errados = [i for i in where(particao==prot)[0] if classesTreinamento[i] != classesPrototipos[prot]]
            distancias[errados,:] = 0
            distancias[where(particao!=prot)[0],prot] = 0
    #somatoriosClasses = array([])
    #for classe in range(nClasses):
    #    somatoriosClasses = append(somatoriosClasses, sum(distancias[:,where(classesPrototipos == classe)[0]]))
    #if any(somatoriosClasses == 0):
    #    pesosClasses = ones(nClasses)
    #else:
    #    pesosClasses = (prod(somatoriosClasses)**(1.0/nClasses))/somatoriosClasses
    pesosPrototipos = ones(k)
    for classe in range(nClasses):
        somatorioPrototipos = sum(distancias[:,where(classesPrototipos == classe)[0]],axis=0)
        protsClasse = where(classesPrototipos == classe)[0]
        if any(somatorioPrototipos == 0):
            pesosPrototipos[protsClasse] = ones(size(protsClasse)) #* pesosClasses[classe]**(1.0/size(protsClasse)) 
        else:
            pesosPrototipos[protsClasse] = ((prod(somatorioPrototipos))**(1.0/size(protsClasse)))/somatorioPrototipos
    
    return pesos * (pesosPrototipos.reshape((k,1))**(1.0/sum(consideradas)))

def treinar(mins, maxs, particulas, velocidades, classesParticulas, PBEST, criteriosPBEST, GBEST, classesGBEST, criterioGBEST, treinamento, classesTreinamento, np, intervalar = True):
    p = shape(treinamento)[1]/2
    if not intervalar:
        p = shape(treinamento)[1]
    k = shape(GBEST)[0]-1
    mi = 100
    w_max = 0.9 
    w_min = 0.4
    w = ones((np,1))*0.9
    c1 = ones((np,1))*2
    c2 = ones((np,1))*2
           
    MAX_ITERACAO_TOTAL = 1000
    epsilon = 0.00001
    Jatual = 1
    Jdepois = -1
    r = 0
    manteve = 0
    while r < MAX_ITERACAO_TOTAL and manteve < 50:
        for k in range(np):
            #Atualizacao da velocidade e posicao
            r1 = random.rand()
            r2 = random.rand()
            velocidades[k] = w[k] * velocidades[k] + c1[k]*r1*(PBEST[k] - particulas[k]) + c2[k]*r2*(GBEST - particulas[k])

            particulas[k] = ajustarMinsMaxs(ajustarVariaveisConsideradas(particulas[k] + velocidades[k],p), mins, maxs, intervalar)
            fi = calcularFi(particulas[k], PBEST[k], GBEST)
            if fi > 3.:
                fi = 3.
            if fi > -1:
                w[k] = ((w_max - w_min)/(1 + exp(fi*(r-((1+log(fi))*MAX_ITERACAO_TOTAL)/mi)))) + w_min
                c1[k] = c1[k]*(fi**-1)
                c2[k] = c2[k]*fi  
            [novoPBEST, novoCriterioPBEST, GBEST, criterioGBEST, classesGBEST] = atualizarMelhores(particulas[k], classesParticulas[k], PBEST[k], criteriosPBEST[k], GBEST, criterioGBEST, classesGBEST, treinamento, classesTreinamento, p, intervalar)
            PBEST[k] = novoPBEST
            criteriosPBEST[k] = novoCriterioPBEST
        Jatual = Jdepois     
        Jdepois = criterioGBEST
        if abs(Jatual - Jdepois) <= epsilon:
            manteve = manteve + 1
        else:
            manteve = 0
        if manteve == 50:
            pesos = calcularPesosPrototipos(treinamento, classesTreinamento, GBEST[1:], classesGBEST, GBEST[0,:p], shape(treinamento)[0], shape(GBEST)[0]-1, p, intervalar)    
            novoGBEST = refinar(treinamento, classesTreinamento, copy(GBEST), classesGBEST, pesos, intervalar, 0.3)
            pesos = calcularPesosPrototipos(treinamento, classesTreinamento, novoGBEST[1:], classesGBEST, novoGBEST[0,:p], shape(treinamento)[0], shape(novoGBEST)[0]-1, p, intervalar)    
            criterio = calcularCriterioJ(pesos, novoGBEST, classesGBEST, treinamento, classesTreinamento, p, intervalar)
            if criterio < Jdepois:
                Jdepois = criterio
                manteve = 0
                GBEST = copy(novoGBEST)
                criterioGBEST = criterio
        r = r+1
    return [GBEST, classesGBEST]    

def refinar(treinamento, classesTreinamento, particula, classesParticula, pesos, intervalar = True, taxa = 0.3):
    k = shape(particula)[0]-1
    n = shape(treinamento)[0]
    taxas = ones(k) * taxa
    passos = 0
    parar = 0
    quantosPararam = 0
    p = shape(treinamento)[1]/2
    if not intervalar:
        p = shape(treinamento)[1]    
    
    prototipos = copy(particula[1:])
    consideradas = around(particula[0,:p])
    
    while parar < 3 and passos < 500:
        i = random.choice(n)
        ds = treinamento[i,:]
        if intervalar:
            mins = consideradas * pesos * ((prototipos[:,::2] - ds[::2])**2)
            maxs = consideradas * pesos * ((prototipos[:,1::2] - ds[1::2])**2)
            dists = sum(mins + maxs,1) + 0.0000000001
        else:
            dists = sum(consideradas * pesos * ((prototipos - ds)**2),1) + 0.0000000001 
        graus = (dists*sum(1.0/dists,keepdims=True))**-1
        for indice in range (k):
            acertou = -1.0
            if classesTreinamento[i] == classesParticula[indice]:
                acertou = 1.0
            prototipos[indice,:] = prototipos[indice,:] + acertou * graus[indice] * taxas[indice] * (ds - prototipos[indice,:])
            taxas[indice] = taxas[indice]/(1.0 +(acertou*taxas[indice]))
            if ((taxas[indice] != 0) and ((taxas[indice] < 0.0001) or (taxas[indice] >= 1.0))):
                taxas[indice] = 0
                quantosPararam += 1
        if quantosPararam >= k:
            parar = 3
            break
        passos += 1 
    particula[1:] = prototipos   
    return particula

def testar(teste, classesTeste, particula, classesParticula, pesos, intervalar = True):
    k = shape(particula)[0] - 1
    n = shape(teste)[0]
    p = shape(teste)[1]/2
    if not intervalar:
        p = shape(teste)[1]
    distancias = calcularDistancias(particula[1:], teste, pesos, particula[0,:p], n, k, intervalar)
    graus = fcm.calcularGraus(distancias)
    particao = argmax(graus,1)       
    classesResultantes = array([classesParticula[prot] for prot in particao])
    numeroErros = float(size(classesTeste[classesTeste != classesResultantes]))
    return (numeroErros / n)*100.0

def rodarValidacaoCruzada(dados, classes, nProts, montecarlo, nFolds, np, intervalar = True): 
    if len(nProts) == 1:
        nProts = ones(max(classes)+1) * nProts[0] 
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
            [particulas, velocidades, classesParticulas, PBEST, criteriosPBEST, GBEST, classesGBEST, criterioGBEST, mins, maxs] = inicializar(treinamento, classesTreinamento, nProts, np, intervalar)
            [GBEST, classesGBEST]  = treinar(mins, maxs, particulas, velocidades, classesParticulas, PBEST, criteriosPBEST, GBEST, classesGBEST, criterioGBEST, treinamento, classesTreinamento, np, intervalar)
            
            p = shape(teste)[1]/2
            if not intervalar:
                p = shape(teste)[1]
            pesos = calcularPesosPrototipos(treinamento, classesTreinamento, GBEST[1:], classesGBEST, GBEST[0,:p], shape(treinamento)[0], shape(GBEST)[0]-1, p, intervalar)
            erros[i*nFolds + fold] =  testar(teste, classesTeste, GBEST, classesGBEST, pesos, intervalar)
            print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)        

def lerDados(nome, intervalar = True):
    if intervalar:
        pasta = "dados/"
    else:
        pasta = "dadosC/"
    with open(pasta + nome, 'r') as f:
        dados = array([line.split() for line in f])
        dados = dados.astype(float)
        dados = dados[dados[:,shape(dados)[1]-1].argsort()]
        classes = dados[:,shape(dados)[1]-1].astype(int)
        dados = dados[:,0:shape(dados)[1]-1]
    return [dados, classes]

if __name__ == "__main__":
    random.seed(1)
    
    [dados, classes] = lerDados("mediterraneo_oceanico_normalizados.txt", True)
    rodarValidacaoCruzada(dados, classes, [10,22], 10, 10, 10, True)       