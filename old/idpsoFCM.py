from numpy import *

import fastslvq as slvq
import fcm as fcm


#from correlacao import matrizDeCorrelacao

def gerarSolucao(treinamento, classesTreinamento, mins, maxs, nProts, intervalar = True):
    nClasses = classesTreinamento.max()+ 1
    variaveisConsideradas = random.rand(shape(treinamento)[1])    
    prototipos = zeros((sum(nProts),shape(treinamento)[1]))
    classes = zeros(sum(nProts))
    inicio = 0
    fim = 0
    p = shape(treinamento)[1]/2
    if not intervalar:
        p = shape(treinamento)[1]    
    variaveisConsideradas[p:] = 0
    for classe in range(nClasses):
        inicio = fim
        fim = fim + nProts[classe]
        membros = where(classesTreinamento == classe)[0]
        for var in range(p):
            if intervalar:
                minimo = min(treinamento[membros,2*var])
                maximo = max(treinamento[membros,2*var+1])
                prototipos[inicio:fim,2*var] = minimo + random.rand(nProts[classe])*(maximo - minimo)
                prototipos[inicio:fim,2*var+1] = minimo + random.rand(nProts[classe])*(maximo - minimo)
            else:
                prototipos[inicio:fim,var] = minimo + random.rand(nProts[classe])*(maximo - minimo)
        classes[inicio:fim] = classe
    
    fonte = ajustarMinsMaxs(ajustarVariaveisConsideradas(append([variaveisConsideradas],prototipos,axis=0),p), mins, maxs, intervalar)    
    pesos = calcularPesosPrototipos(treinamento, classesTreinamento, fonte[1:], classes, fonte[0,:p], shape(treinamento)[0], shape(fonte)[0]-1, p, intervalar)
    return [fonte, classes, pesos]

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
    pesosParticulas = ones((np,sum(nProts),p))
    velocidades = random.rand(np,sum(nProts)+1,shape(treinamento)[1])
    classesParticulas = zeros((np,sum(nProts)))
    criterios = zeros(np)
    for particula in range(np):
        #variaveisConsideradas = random.rand(shape(treinamento)[1])
        #[prototipos, classesPrototipos] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts)        
        #variaveisConsideradas[p:] = 0
        #particulas[particula] = append([variaveisConsideradas],prototipos,axis=0)
        #classesParticulas[particula] = classesPrototipos
        #particulas[particula] = ajustarMinsMaxs(ajustarVariaveisConsideradas(particulas[particula],p), mins, maxs, intervalar)
        #pesosParticulas[particula] = calcularPesosPrototipos(treinamento, classesTreinamento, particulas[particula,1:], classesParticulas[particula], particulas[particula,0,:p], shape(treinamento)[0], shape(particulas[particula])[0]-1, p, intervalar)
        [particulas[particula], classesParticulas[particula], pesosParticulas[particula]] = gerarSolucao(treinamento, classesTreinamento, mins, maxs, nProts, intervalar)
        if particula == 0:
            [particulas[particula], pesosParticulas[particula]] = refinar(treinamento, classesTreinamento, particulas[particula], classesParticulas[particula], pesosParticulas[particula], intervalar)
        
        
        criterios[particula] = calcularCriterioJ(pesosParticulas[particula], particulas[particula], classesParticulas[particula], treinamento, classesTreinamento, p, intervalar)
    PBEST = copy(particulas)
    criteriosPBEST = copy(criterios)
    pesosPBEST = copy(pesosParticulas)
    indice = argmin(criterios)
    GBEST = copy(particulas[indice])
    classesGBEST = copy(classesParticulas[indice])
    criterioGBEST = criterios[indice]
    pesosGBEST = copy(pesosParticulas[indice])
    return [pesosParticulas, particulas, velocidades, classesParticulas, PBEST, criteriosPBEST, pesosPBEST, GBEST, classesGBEST, criterioGBEST, pesosGBEST, mins, maxs]

def selecionarConsideradas(treinamento, variaveisConsideradas, intervalar = True):
    n = shape(treinamento)[0]    
    consideradas = around(variaveisConsideradas)
    pConsideradas = sum(consideradas)
    if intervalar:
        selecionadas = zeros((n,2*pConsideradas))
    else:
        selecionadas = zeros((n,pConsideradas))
    proxima = 0
    for i,c in enumerate(consideradas):
        if c == 1:
            if intervalar:
                selecionadas[:,proxima] = treinamento[:,i*2]
                selecionadas[:,proxima+1] = treinamento[:,i*2+1]
                proxima = proxima + 2
            else:
                selecionadas[:,proxima] = treinamento[:,i]
                proxima = proxima + 1
    return selecionadas
        

def calcularCriterioJ(pesos, particula, classesParticula, treinamento, classesTreinamento, p, intervalar = True):
    alfa = 0.5
    beta = 0.5
    #gama = 0.2
    
    erro = testar(treinamento, classesTreinamento, particula, classesParticula, pesos, intervalar)
    
    n = shape(treinamento)[0]
    k = shape(particula)[0]-1
    distancias = calcularDistancias(particula[1:], treinamento, pesos, particula[0,:p], n, k, intervalar)
    graus = fcm.calcularGraus(distancias)
    criterio = fcm.calcularCriterio(graus, distancias, classesTreinamento, classesParticula)
    
    #selecionadas = selecionarConsideradas(treinamento, particula[0,:p], intervalar)
    #prodCorrInterna = prod(abs(matrizDeCorrelacao(selecionadas, intervalar)))
    
    return alfa * erro + beta * criterio #+ gama * prodCorrInterna#(erro * criterio) / (sum (particula[0,:p]) + 0.0000000001)
 
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

def atualizarMelhores(pesosParticula, particula, classesParticula, PBEST, criterioPBEST, pesosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, treinamento, classesTreinamento, p, limites, ind, intervalar = True):
    #pesos = calcularPesosPrototipos(treinamento, classesTreinamento, particula[1:], classesParticula, particula[0,:p], shape(treinamento)[0], shape(particula)[0]-1, p, intervalar)
    criterio = calcularCriterioJ(pesosParticula, particula, classesParticula, treinamento, classesTreinamento, p, intervalar)
    if criterio < criterioPBEST:
        criterioPBEST = criterio
        PBEST = copy(particula)
        pesosPBEST = copy(pesosParticula)
        limites[ind] = 0
    else:
        limites[ind] = limites[ind] + 1
    if criterio < criterioGBEST:
        criterioGBEST = criterio
        GBEST = copy(particula)
        classesGBEST = copy(classesParticula)   
        pesosGBEST = copy(pesosParticula)
    return [PBEST, criterioPBEST, pesosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, limites]

def ajustarVariaveisConsideradas(particula,p):
    if p == len(particula[0,:]) / 2:
        particula[0,p:] = zeros(p)
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
    deltas = zeros((k,p))
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
            deltas[prot,:] = sum(grausMembros.reshape((size(grausMembros),1)) * consideradas * diff,0)
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
            pesosPrototipos[protsClasse] = ((pesosClasses[classe]* prod(somatorioPrototipos))**(1.0/size(protsClasse)))/somatorioPrototipos
        for prot in protsClasse:
            achou = False
            index = 0
            delta = deltas[prot,:]
            while achou == False and index < p:
                if consideradas[index] == 1.0 and delta[index] == 0.0:
                    achou = True
                index = index + 1
            if achou == False:
                produtorio = (pesosPrototipos[prot]*prod(delta[delta > 0]))**(1.0/sum(consideradas))
                pesos[prot,:] = array([produtorio / v if v > 0 else 0 for v in delta])
            else:
                pesos[prot,:] = array([pesosPrototipos[prot]**(1.0/sum(consideradas)) if v > 0 else 0 for v in consideradas])
    return pesos

def melhorarPBEST(np, particulas, pesosParticulas, PBEST, classesParticulas, pesosPBEST, criteriosPBEST, mins, maxs, treinamento, classesTreinamento, limites, limite, nProts, GBEST, criterioGBEST, classesGBEST, pesosGBEST, intervalar = True):
    ind = argmax(limites)
    if limites[ind] >= limite:     
        p = shape(treinamento)[1]/2
        if not intervalar:
            p = shape(treinamento)[1]   
        [va, pesos] = refinar(treinamento, classesTreinamento, PBEST[ind], classesParticulas[ind], pesosPBEST[ind], intervalar)
        criterio = calcularCriterioJ(pesos, va, classesParticulas[ind], treinamento, classesTreinamento, p, intervalar)
        if criterio < criteriosPBEST[ind]:
            criteriosPBEST[ind] = criterio
            limites[ind] = 0
            PBEST[ind] = va
            pesosPBEST[ind] = pesos
            particulas[ind] = va
            pesosParticulas[ind] = pesos
        else:
            [particulas[ind], classesParticulas[ind], pesosParticulas[ind]] = gerarSolucao(treinamento, classesTreinamento, mins, maxs, nProts, intervalar)
            limites[ind] = 0
            
            criterio = calcularCriterioJ(pesosParticulas[ind], particulas[ind], classesParticulas[ind], treinamento, classesTreinamento, p, intervalar)
            if criterio < criteriosPBEST[ind]:
                criteriosPBEST[ind] = criterio
                PBEST[ind] = copy(particulas[ind])
                pesosPBEST[ind] = copy(pesosParticulas[ind])
        if criteriosPBEST[ind] < criterioGBEST:
            criterioGBEST = criteriosPBEST[ind]    
            GBEST = copy(PBEST[ind])
            classesGBEST = classesParticulas[ind]
            pesosGBEST = copy(pesosPBEST[ind])
    return [particulas, pesosParticulas, classesParticulas, PBEST, pesosPBEST, criteriosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, limites]

def fazerCrossover(np, particulas, pesosParticulas, PBEST, classesParticulas, pesosPBEST, criteriosPBEST, mins, maxs, treinamento, classesTreinamento, limites, GBEST, criterioGBEST, classesGBEST, pesosGBEST, intervalar = True):
    nPais = int(round(np / 2))
    pa = shape(treinamento)[1]/2
    if not intervalar:
        pa = shape(treinamento)[1]
    k = shape(particulas[0])[0] - 1
    n = shape(treinamento)[0]
    pais  = zeros((nPais, k+1, shape(treinamento)[1]))
    fPais = zeros(nPais)
    probs = (1.0/criteriosPBEST) / sum(1.0/criteriosPBEST)
    for i in range (nPais):        
        k1 = random.choice(np,p=probs)
        k2 = random.choice(np,p=probs)
        if criteriosPBEST[k1] < criteriosPBEST[k2]:
            pais[i] = PBEST[k1]
            fPais[i] = criteriosPBEST[k1]
        else:
            pais[i] = PBEST[k2]
            fPais[i] = criteriosPBEST[k2]
    pPais = (1.0/fPais) / sum(1.0/fPais)
    for i in range(np):
        genitor1 = random.choice(nPais,p=pPais)
        genitor2 = genitor1
        while genitor2 == genitor1:
            genitor2 = random.choice(nPais,p=pPais)
            temp = random.rand()
        prole = temp* pais[genitor1] + (1-temp)* pais[genitor2]
        
        prole = ajustarMinsMaxs(ajustarVariaveisConsideradas(prole,pa), mins, maxs, intervalar)
        
        pesos = calcularPesosPrototipos(treinamento, classesTreinamento, prole[1:], classesParticulas[i], prole[0,:pa], n, k, pa, intervalar)
            
        criterio = calcularCriterioJ(pesos, prole, classesParticulas[i], treinamento, classesTreinamento, pa, intervalar)
        
        if criterio < criteriosPBEST[i]:
            criteriosPBEST[i] = criterio
            limites[i] = 0
            PBEST[i] = copy(prole)
            pesosPBEST[i] = copy(pesos)
        if criteriosPBEST[i] < criterioGBEST:
            criterioGBEST = criteriosPBEST[i]    
            GBEST = copy(prole)
            classesGBEST = classesParticulas[i]
            pesosGBEST = copy(pesos)
    return [particulas, pesosParticulas, classesParticulas, PBEST, pesosPBEST, criteriosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, limites]

def treinar(mins, maxs, particulas, velocidades, classesParticulas, pesosParticulas, PBEST, criteriosPBEST, pesosPBEST, GBEST, classesGBEST, criterioGBEST, pesosGBEST, treinamento, classesTreinamento, np, nProts, intervalar = True):
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
    
    limite = 10
    limites = zeros(np)
    
    while r < MAX_ITERACAO_TOTAL and manteve < 50:
        for k in range(np):
            #if r == 313 and k == 18:
            #    print k
            #Atualizacao da velocidade e posicao
            r1 = random.rand()
            r2 = random.rand()
            velocidades[k] = w[k] * velocidades[k] + c1[k]*r1*(PBEST[k] - particulas[k]) + c2[k]*r2*(GBEST - particulas[k])
            consideradasAntes = copy(particulas[k,0,:p])
            particulas[k] = ajustarMinsMaxs(ajustarVariaveisConsideradas(particulas[k] + velocidades[k],p), mins, maxs, intervalar)
            pe = pesosParticulas[k]
            if any(around(consideradasAntes) != around(particulas[k,0,:p])):
                pe = []
            pesosParticulas[k] = calcularPesosPrototipos(treinamento, classesTreinamento, particulas[k,1:], classesParticulas[k], particulas[k,0,:p], shape(treinamento)[0], shape(particulas[k])[0]-1, p, intervalar, pe)
            fi = calcularFi(particulas[k], PBEST[k], GBEST)
            if fi > -1:
                w[k] = ((w_max - w_min)/(1 + exp(fi*(r-((1+log(fi))*MAX_ITERACAO_TOTAL)/mi)))) + w_min
                c1[k] = c1[k]*(fi**-1)
                c2[k] = c2[k]*fi  
            [novoPBEST, novoCriterioPBEST, novosPesosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, limites] = atualizarMelhores(pesosParticulas[k], particulas[k], classesParticulas[k], PBEST[k], criteriosPBEST[k], pesosPBEST[k], GBEST, criterioGBEST, classesGBEST, pesosGBEST, treinamento, classesTreinamento, p, limites, k, intervalar)
            PBEST[k] = novoPBEST
            criteriosPBEST[k] = novoCriterioPBEST
            pesosPBEST[k] = novosPesosPBEST
        [particulas, pesosParticulas, classesParticulas, PBEST, pesosPBEST, criteriosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, limites] = fazerCrossover(np, particulas, pesosParticulas, PBEST, classesParticulas, pesosPBEST, criteriosPBEST, mins, maxs, treinamento, classesTreinamento, limites, GBEST, criterioGBEST, classesGBEST, pesosGBEST, intervalar)
        [particulas, pesosParticulas, classesParticulas, PBEST, pesosPBEST, criteriosPBEST, GBEST, criterioGBEST, classesGBEST, pesosGBEST, limites] = melhorarPBEST(np, particulas, pesosParticulas, PBEST, classesParticulas, pesosPBEST, criteriosPBEST, mins, maxs, treinamento, classesTreinamento, limites, limite, nProts, GBEST, criterioGBEST, classesGBEST, pesosGBEST, intervalar)
        fi = calcularFi(particulas[k], PBEST[k], GBEST)
        if fi > -1:
            w[k] = ((w_max - w_min)/(1 + exp(fi*(r-((1+log(fi))*MAX_ITERACAO_TOTAL)/mi)))) + w_min
            c1[k] = c1[k]*(fi**-1)
            c2[k] = c2[k]*fi  
        Jatual = Jdepois     
        Jdepois = criterioGBEST
        if abs(Jatual - Jdepois) <= epsilon:
            manteve = manteve + 1
        else:
            manteve = 0
        if manteve == 50:
            [novoGBEST, pesos] = refinar(treinamento, classesTreinamento, copy(GBEST), classesGBEST, pesosGBEST, intervalar)
            criterio = calcularCriterioJ(pesos, novoGBEST, classesGBEST, treinamento, classesTreinamento, p, intervalar)
            if criterio < Jdepois:
                Jdepois = criterio
                manteve = 0
                GBEST = copy(novoGBEST)
                criterioGBEST = criterio
                pesosGBEST = copy(pesos)
        r = r+1
    return [GBEST, classesGBEST, pesosGBEST]    

def refinar(treinamento, classesTreinamento, particula, classesParticula, pesosP, intervalar = True):
    t = 1
    tMax = 50
    p = shape(treinamento)[1]/2
    if not intervalar:
        p = shape(treinamento)[1]    
    
    prototipos = copy(particula[1:])
    variaveisConsideradas = particula[0,:p]
    pesos = copy(pesosP)
    
    k = shape(prototipos)[0]
    n = shape(treinamento)[0]
    epsilon = 0.00001
    Jatual = 1
    Jdepois = -1
    while t <= tMax and abs(Jatual - Jdepois)>epsilon:
        #fase dos prototipos
        distancias = calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k, intervalar)
        graus = fcm.calcularGraus(distancias)  
                
        Jatual = Jdepois
        Jdepois = fcm.calcularCriterio(copy(graus), distancias, classesTreinamento, classesParticula)
                  
        particao = argmax(graus,1)        
        prototipos = fcm.calcularPrototipos(graus, treinamento, classesTreinamento, prototipos, classesParticula, particao, k)
                
        t = t + 1
        #fase dos pesos
        pesos = calcularPesosPrototipos(treinamento, classesTreinamento, prototipos, classesParticula, variaveisConsideradas, n, k, p, intervalar, pesos)    
    particula[1:] = prototipos
    return [particula, pesos]

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
    p = shape(dados)[1]/2
    if not intervalar:
        p = shape(dados)[1]
    consideradas = zeros(p)
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
            [pesosParticulas, particulas, velocidades, classesParticulas, PBEST, criteriosPBEST, pesosPBEST, GBEST, classesGBEST, criterioGBEST, pesosGBEST, mins, maxs] = inicializar(treinamento, classesTreinamento, nProts, np, intervalar)
            [GBEST, classesGBEST, pesosGBEST] = treinar(mins, maxs, particulas, velocidades, classesParticulas, pesosParticulas, PBEST, criteriosPBEST, pesosPBEST, GBEST, classesGBEST, criterioGBEST, pesosGBEST, treinamento, classesTreinamento, np, nProts, intervalar)
            consideradas = consideradas + GBEST[0,:p]
            erros[i*nFolds + fold] =  testar(teste, classesTeste, GBEST, classesGBEST, pesosGBEST, intervalar)
            print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo * nFolds)
    
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
    print "melhorarPBESTpso0.50.5"
    print "mediterraneo_oceanico_normalizados_limpo.txt"
    [dados, classes] = lerDados("mediterraneo_oceanico_normalizados_limpo.txt", True)
    rodarValidacaoCruzada(dados, classes, [20, 40], 10, 10, 20, True)      