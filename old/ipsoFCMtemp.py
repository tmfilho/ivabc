from numpy import *

import fastslvq as slvq
import fcm as fcm


def gerarSolucao(nProts):
    #nClasses = classesTreinamento.max()+ 1
    variaveisConsideradas = random.rand(shape(treinamento)[1])    
    [prototipos,classes] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts)
    #prototipos = zeros((sum(nProts),shape(treinamento)[1]))
    #classes = zeros(sum(nProts))
    #inicio = 0
    #fim = 0  
    #variaveisConsideradas[pa:] = 0
    #for classe in range(nClasses):
    #    inicio = fim
    #    fim = fim + nProts[classe]
    #    membros = where(classesTreinamento == classe)[0]
    #    for var in range(pa):
    #        if intervalar:
    #            minimo = min(treinamento[membros,2*var])
    #            maximo = max(treinamento[membros,2*var+1])
    #            prototipos[inicio:fim,2*var] = minimo + random.rand(nProts[classe])*(maximo - minimo)
    #            prototipos[inicio:fim,2*var+1] = minimo + random.rand(nProts[classe])*(maximo - minimo)
    #        else:
    #            prototipos[inicio:fim,var] = minimo + random.rand(nProts[classe])*(maximo - minimo)
    #    classes[inicio:fim] = classe
    
    fonte = ajustarMinsMaxs(ajustarVariaveisConsideradas(append([variaveisConsideradas],prototipos,axis=0)))    
    pesos = calcularPesosPrototipos(fonte[1:], classes, fonte[0,:pa])
    return [fonte, classes, pesos]

def iniciarVelocidades(np):
    global velocidades
    if intervalar:
        velocidades = random.uniform(-1,1,(np,k+1,2*pa))
        for var in range(pa):
            if (maxs[var]-mins[var]) > 1:
                velocidades[:,1:,2*var] = velocidades[:,1:,2*var] * (maxs[var]-mins[var])
                velocidades[:,1:,2*var+1] = velocidades[:,1:,2*var+1] * (maxs[var]-mins[var])
    else:
        for var in range(pa):
            if (maxs[var]-mins[var]) > 1:
                velocidades[:,1:,var] = velocidades[:,1:,var] * (maxs[var]-mins[var])

def iniciarVelocidade():
    if intervalar:
        velocidade = random.uniform(-1,1,(k+1,2*pa))
        for var in range(pa):
            if (maxs[var]-mins[var]) > 1:
                velocidade[1:,2*var] = velocidade[1:,2*var] * (maxs[var]-mins[var])
                velocidade[1:,2*var+1] = velocidade[1:,2*var+1] * (maxs[var]-mins[var])
    else:
        for var in range(pa):
            if (maxs[var]-mins[var]) > 1:
                velocidade[1:,var] = velocidades[1:,var] * (maxs[var]-mins[var])
    return velocidade

def ajustarVelocidade(velocidade): 
    consideradas = velocidade[0,:]
    consideradas[consideradas > 1] = 1
    consideradas[consideradas < -1] = -1
    velocidade[0,:] = consideradas
    if intervalar:
        for var in range(pa):
            minimos = velocidade[1:,2*var]
            minimos[minimos < (-1 * (maxs[var]-mins[var]))] = (-1 * (maxs[var]-mins[var]))
            minimos[minimos > (maxs[var]-mins[var])] = (maxs[var]-mins[var])  
            velocidade[1:,2*var] = minimos
            maximos = velocidade[1:,2*var+1]
            maximos[maximos < (-1 * (maxs[var]-mins[var]))] = (-1 * (maxs[var]-mins[var]))
            maximos[maximos > (maxs[var]-mins[var])] = (maxs[var]-mins[var])   
            velocidade[1:,2*var+1] = maximos         
    else:
        for var in range(pa):
            temp = velocidade[1:,var]
            temp[temp < (-1 * (maxs[var]-mins[var]))] = (-1 * (maxs[var]-mins[var]))
            temp[temp > (maxs[var]-mins[var])] = (maxs[var]-mins[var]) 
            velocidade[1:,var] = temp
    return velocidade
    #velocidade[velocidade < -1.0] = -1.0
    #velocidade[velocidade > 1.0] = 1.0
    #return velocidade

def inicializar(nProts, np):
    global particulas
    global pesosParticulas
    global classesParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global GBEST
    global classesGBEST
    global criterioGBEST
    global pesosGBEST
    global mins
    global maxs
    global criterios
    #global medias
    
    #[medias, classesTemp] = slvq.iniciarPrototiposPelaMedia(treinamento, classesTreinamento)
    
    if intervalar:
        mins = treinamento[:,::2].min(0)
        maxs = treinamento[:,1::2].max(0)
    else:
        mins = treinamento.min(0)
        maxs = treinamento.max(0)
    particulas = zeros((np,sum(nProts)+1,shape(treinamento)[1]))
    pesosParticulas = ones((np,sum(nProts),pa))
    iniciarVelocidades(np)
    classesParticulas = zeros((np,sum(nProts)))
    criterios = zeros(np)
    for particula in range(np):
        [particulas[particula], classesParticulas[particula], pesosParticulas[particula]] = gerarSolucao(nProts)
        #if particula == 0:
        #    [particulas[particula], pesosParticulas[particula]] = refinar(particulas[particula], classesParticulas[particula], pesosParticulas[particula])
              
        criterios[particula] = calcularCriterioJ(particulas[particula], pesosParticulas[particula], classesParticulas[particula])
    PBEST = copy(particulas)
    criteriosPBEST = copy(criterios)
    pesosPBEST = copy(pesosParticulas)
    indice = argmin(criterios)
    GBEST = copy(particulas[indice])
    classesGBEST = copy(classesParticulas[indice])
    criterioGBEST = criterios[indice]
    pesosGBEST = copy(pesosParticulas[indice])     

def calcularCriterioJ(particula, pesos, classesParticula):
    erro = testar(treinamento, classesTreinamento, particula, classesParticula, pesos)
        
    #distancias = calcularDistancias(particula[1:], treinamento, pesos, particula[0,:pa], n, k)
    #graus = fcm.calcularGraus(distancias)
    #criterio = fcm.calcularCriterio(graus, distancias, classesTreinamento, classesParticula)
        
    #return alfa * erro + beta * criterio
    
    nClasses = max(classesTreinamento)+1
    medias = zeros((nClasses, shape(particula)[1]))
    classesTemp = arange(nClasses)
    prots = particula[1:]
    for classe in classesTemp:
        protsClasse = prots[classesParticula==classe]
        medias[classe] = sum(protsClasse,0)/shape(protsClasse)[0]
    distanciasProtCentros = calcularDistancias(particula[1:], medias, pesos, particula[0,:pa], nClasses, k)
    distanciasMembrosCentros = calcularDistancias(medias, treinamento, ones((nClasses,pa)), particula[0,:pa], n, nClasses)
    for classe in classesTemp:
        distanciasMembrosCentros[classe!=classesTreinamento,classe] = 0.0
    for prot in range(k):
        distanciasProtCentros[classesTemp!=classesParticula[prot],prot] = 0.0
        #distanciasMembrosProts[classesTreinamento!=classesParticula[prot],prot] = 0.0
        #distanciasMembrosProts[logical_and(particao==prot,classesTreinamento!=classesParticula[prot]),:] = 0.0
    
    return alfa * erro + beta * (sum(distanciasMembrosCentros) / sum(distanciasProtCentros))
    
def calcularDistancias(prototipos, dados, pesos, variaveisConsideradas, n, k):
    consideradas = around(variaveisConsideradas)
    distancias = zeros((n,k))
    for prot in range(k):
        if intervalar:
            mins = consideradas * pesos[prot,:] * ((prototipos[prot,::2] - dados[:,::2])**2)
            maxs = consideradas * pesos[prot,:] * ((prototipos[prot,1::2] - dados[:,1::2])**2)
            distancias[:,prot] = sum(mins + maxs,1) + 0.0000000001
        else:
            distancias[:,prot] = sum((consideradas * pesos[prot,:] * (prototipos[prot,:] - dados)**2),1) + 0.0000000001
    return distancias

def calcularFi(indice):
    particula = particulas[indice]
    pbest = PBEST[indice]
    fi = -1
    numerador = sqrt(sum((GBEST-particula)**2.0))
    denominador = sqrt(sum((pbest-particula)**2.0))
    if numerador != 0.0 and denominador != 0.0:
        fi = numerador/denominador
    if fi > 10:
        fi = 10
    return fi

def atualizarMelhores(indice):
    global criteriosPBEST
    global PBEST
    global pesosPBEST
    global limites
    global criterioGBEST
    global GBEST
    global classesGBEST
    global pesosGBEST
    global criterios
    
    criterio = calcularCriterioJ(particulas[indice], pesosParticulas[indice], classesParticulas[indice])
    criterios[indice] = criterio
    if criterio < criteriosPBEST[indice]:
        criteriosPBEST[indice] = criterio
        PBEST[indice] = copy(particulas[indice])
        pesosPBEST[indice] = copy(pesosParticulas[indice])
        limites[indice] = 0
    else:
        limites[indice] = limites[indice] + 1
    if criterio < criterioGBEST:
        criterioGBEST = criterio
        GBEST = copy(particulas[indice])
        classesGBEST = copy(classesParticulas[indice])   
        pesosGBEST = copy(pesosParticulas[indice])

def ajustarVariaveisConsideradas(particula):
    if pa == len(particula[0,:]) / 2:
        particula[0,pa:] = zeros(pa)
    consideradas = particula[0,:pa]
    consideradas[consideradas < 0] = 0
    consideradas[consideradas > 1] = 1
    while all(around(consideradas) == 0):
        consideradas = random.rand(len(consideradas))
    particula[0,:pa] = consideradas
    return particula

def ajustarMinsMaxs(dados):
    #prots = dados[1:]
    #prots[prots < 0.0] = 0.0
    #prots[prots > 1.0] = 1.0
    #dados[1:] = prots
    if intervalar:
        for var in range(pa):
            minimos = dados[1:,2*var]
            minimos[minimos < mins[var]] = mins[var]
            minimos[minimos > maxs[var]] = maxs[var]
            maximos = dados[1:,2*var+1]
            maximos[maximos < mins[var]] = mins[var]
            maximos[maximos > maxs[var]] = maxs[var] 
            
            ind = minimos > maximos
            temp = minimos[ind]
            minimos[ind] = maximos[ind]
            maximos[ind] = temp
                
            dados[1:,2*var] = minimos
            dados[1:,2*var+1] = maximos         
    else:
        for var in range(pa):
            temp = dados[1:,var]
            temp[temp < mins[var]] = mins[var]
            temp[temp > maxs[var]] = maxs[var] 
            dados[1:,var] = temp
    return dados

def calcularPesosPrototipos(prototipos, classesPrototipos, variaveisConsideradas, pesos = []):
    nClasses = max(classesTreinamento + 1)
    consideradas = around(variaveisConsideradas)
    if len(pesos) == 0:
        pesos = ones((k,pa))
    distancias = calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k)
    graus = fcm.calcularGraus(distancias)  
    
    particao = argmax(graus,1)
    distancias = (graus**2)*distancias
    deltas = zeros((k,pa))
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
            while achou == False and index < pa:
                if consideradas[index] == 1.0 and delta[index] == 0.0:
                    achou = True
                index = index + 1
            if achou == False:
                produtorio = (pesosPrototipos[prot]*prod(delta[delta > 0]))**(1.0/sum(consideradas))
                pesos[prot,:] = array([produtorio / v if v > 0 else 0 for v in delta])
            else:
                pesos[prot,:] = array([pesosPrototipos[prot]**(1.0/sum(consideradas)) if v > 0 else 0 for v in consideradas])
    return pesos

def melhorarPBEST(np, limite, nProts):
    global particulas
    global pesosParticulas
    global classesParticulas
    global PBEST
    global pesosPBEST
    global criteriosPBEST
    global GBEST
    global criterioGBEST
    global classesGBEST
    global pesosGBEST
    global limites
    global criterios
    global velocidades
     
    ind = argmax(limites)
    if limites[ind] >= limite:     
        #[va, pesos] = refinar(PBEST[ind], classesParticulas[ind], pesosPBEST[ind])
        #criterio = calcularCriterioJ(va, pesos, classesParticulas[ind])
        #if criterio < criteriosPBEST[ind]:
        #    criteriosPBEST[ind] = criterio
        #    limites[ind] = 0
        #    PBEST[ind] = va
        #    pesosPBEST[ind] = pesos
        #    particulas[ind] = va
        #    pesosParticulas[ind] = pesos
        #    criterios[ind] = criterio
        #else:
        [particulas[ind], classesParticulas[ind], pesosParticulas[ind]] = gerarSolucao(nProts)
        limites[ind] = 0
        
        criterio = calcularCriterioJ(particulas[ind], pesosParticulas[ind], classesParticulas[ind])
        criterios[ind] = criterio
        if criterio < criteriosPBEST[ind]:
            criteriosPBEST[ind] = criterio
            PBEST[ind] = copy(particulas[ind])
            pesosPBEST[ind] = copy(pesosParticulas[ind])
        if criteriosPBEST[ind] < criterioGBEST:
            criterioGBEST = criteriosPBEST[ind]    
            GBEST = copy(PBEST[ind])
            classesGBEST = classesParticulas[ind]
            pesosGBEST = copy(pesosPBEST[ind])
        velocidades[ind] = iniciarVelocidade()
    
def fazerCrossover(np):
    global particulas
    global pesosParticulas
    global PBEST
    global pesosPBEST
    global criteriosPBEST
    global GBEST
    global criterioGBEST
    global classesGBEST
    global pesosGBEST
    global limites
    global criterios
    global velocidades
    
    probs = (1.0/criterios) / sum(1.0/criterios)
    for i in range(np/2):
        pai1 = random.choice(np,p=probs)
        pai2 = pai1
        while pai2 == pai1:
            pai2 = random.choice(np,p=probs)
        #pontoDeCorte = random.choice(arange(k)+1)
        #prole = zeros(shape(particulas[pai1]))
        #prole[:pontoDeCorte] = particulas[pai1,:pontoDeCorte]
        #prole[pontoDeCorte:] = particulas[pai2,pontoDeCorte:]
        temp = random.rand()
        prole = temp * particulas[pai1] + (1 - temp) * particulas[pai2]
        
        prole = ajustarMinsMaxs(ajustarVariaveisConsideradas(prole))
        pesos = calcularPesosPrototipos(prole[1:], classesParticulas[i], prole[0,:pa])            
        criterio = calcularCriterioJ(prole, pesos, classesParticulas[i])
        piorCriterio = argmax(criterios)
        if criterio < criterios[piorCriterio]:
            particulas[piorCriterio] = prole
            pesosParticulas[piorCriterio] = pesos
            criterios[piorCriterio] = criterio
            velocidades[piorCriterio] = iniciarVelocidade()
            if criterio < criteriosPBEST[piorCriterio]:
                criteriosPBEST[piorCriterio] = criterio
                limites[piorCriterio] = 0
                PBEST[piorCriterio] = copy(prole)
                pesosPBEST[piorCriterio] = copy(pesos)
            else:
                limites[piorCriterio] = limites[piorCriterio] + 1
            if criteriosPBEST[piorCriterio] < criterioGBEST:
                criterioGBEST = criteriosPBEST[piorCriterio]    
                GBEST = copy(prole)
                classesGBEST = classesParticulas[piorCriterio]
                pesosGBEST = copy(pesos)
                
                
def fazerCrossover2(np):
    global particulas
    global pesosParticulas
    global PBEST
    global pesosPBEST
    global criteriosPBEST
    global GBEST
    global criterioGBEST
    global classesGBEST
    global pesosGBEST
    global limites
    global criterios
    global velocidades
    
    nPais = int(round(np / 2))
    pais  = zeros((nPais, k+1, shape(treinamento)[1]))
    fPais = zeros(nPais)
    #print "criterios",criterios
    probs = (1.0/criterios) / sum(1.0/criterios)
    #print "probs", probs
    for i in range (nPais):        
        k1 = random.choice(np,p=probs)
        k2 = random.choice(np,p=probs)
        if criterios[k1] < criterios[k2]:
            pais[i] = particulas[k1]
            fPais[i] = criterios[k1]
        else:
            pais[i] = particulas[k2]
            fPais[i] = criterios[k2]
    pPais = (1.0/fPais) / sum(1.0/fPais)
    for i in range(np):
        genitor1 = random.choice(nPais,p=pPais)
        genitor2 = genitor1
        while genitor2 == genitor1:
            genitor2 = random.choice(nPais,p=pPais)
        temp = random.rand()
        prole = temp * pais[genitor1] + (1 - temp) * pais[genitor2]
        
        prole = ajustarMinsMaxs(ajustarVariaveisConsideradas(prole))
        
        pesos = calcularPesosPrototipos(prole[1:], classesParticulas[i], prole[0,:pa])
            
        criterio = calcularCriterioJ(prole, pesos, classesParticulas[i])
        if criterio < criterios[i]:
            particulas[i] = copy(prole)
            pesosParticulas[i] = copy(pesos)
            criterios[i] = criterio   
            velocidades[i] = iniciarVelocidade()
            if criterio < criteriosPBEST[i]:
                criteriosPBEST[i] = criterio
                limites[i] = 0
                PBEST[i] = copy(prole)
                pesosPBEST[i] = copy(pesos)
            else:
                limites[i] = limites[i] + 1
            if criteriosPBEST[i] < criterioGBEST:
                criterioGBEST = criteriosPBEST[i]    
                GBEST = copy(prole)
                classesGBEST = classesParticulas[i]
                pesosGBEST = copy(pesos)

def treinar(np, nProts):
    seterr(all='raise')
    global limites
    global particulas
    global velocidades
    global pesosParticulas
    global GBEST
    global classesGBEST
    global criterioGBEST
    global pesosGBEST
    
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
        #print "r",r
        #print "manteve", manteve
        for particula in range(np):
            #print "particula",particula
            try:
                if r > 0:
                    fi = calcularFi(particula)
                    if fi > -1:
                        temp = fi*(r-((1+log(fi))*MAX_ITERACAO_TOTAL)/mi)
                        if temp > 700:
                            w[particula] = w_min
                        else:
                            w[particula] = ((w_max - w_min)/(1 + exp(temp))) + w_min
                        c1[particula] = c1[particula]*(fi**-1)
                        if c1[particula] > 500.0:
                            c1[particula] = 500.0
                        elif c1[particula] < 1.0e-6:
                            c1[particula] = 0.0
                        c2[particula] = c2[particula]*fi 
                        if c2[particula] > 500.0:
                            c2[particula] = 500.0
                        elif c2[particula] < 1.0e-6:
                            c2[particula] = 0.0
                r1 = random.rand()
                r2 = random.rand()
                velocidades[particula] = ajustarVelocidade(w[particula] * velocidades[particula] + c1[particula]*r1*(PBEST[particula] - particulas[particula]) + c2[particula]*r2*(GBEST - particulas[particula]))
                consideradasAntes = copy(particulas[particula,0,:pa])
                particulas[particula] = ajustarMinsMaxs(ajustarVariaveisConsideradas(particulas[particula] + velocidades[particula]))
                pe = pesosParticulas[particula]
                if any(around(consideradasAntes) != around(particulas[particula,0,:pa])):
                    pe = []
                pesosParticulas[particula] = calcularPesosPrototipos(particulas[particula,1:], classesParticulas[particula], particulas[particula,0,:pa], pe)
                atualizarMelhores(particula)
            except FloatingPointError:
                print "opa"
        fazerCrossover(np)
        melhorarPBEST(np, limite, nProts)
         
        Jatual = Jdepois     
        Jdepois = criterioGBEST
        #print Jdepois
        if abs(Jatual - Jdepois) <= epsilon:
            manteve = manteve + 1
        else:
            manteve = 0
        #if manteve == 50:
        #    [novoGBEST, pesos] = refinar(copy(GBEST), classesGBEST, pesosGBEST)
        #    criterio = calcularCriterioJ(novoGBEST, pesos, classesGBEST)
        #    if criterio < Jdepois:
        #        Jdepois = criterio
        #        manteve = 0
        #        GBEST = copy(novoGBEST)
        #        criterioGBEST = criterio
        #        pesosGBEST = copy(pesos)
        r = r+1

def refinar(particula, classesParticula, pesosP):
    t = 1
    tMax = 50    
    
    prototipos = copy(particula[1:])
    variaveisConsideradas = particula[0,:pa]
    pesos = copy(pesosP)
    
    epsilon = 0.00001
    Jatual = 1
    Jdepois = -1
    while t <= tMax and abs(Jatual - Jdepois)>epsilon:
        #fase dos prototipos
        distancias = calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k)
        graus = fcm.calcularGraus(distancias)  
                
        Jatual = Jdepois
        Jdepois = fcm.calcularCriterio(copy(graus), distancias, classesTreinamento, classesParticula)
                  
        particao = argmax(graus,1)        
        prototipos = fcm.calcularPrototipos(graus, treinamento, classesTreinamento, prototipos, classesParticula, particao, k)
                
        t = t + 1
        #fase dos pesos
        pesos = calcularPesosPrototipos(prototipos, classesParticula, variaveisConsideradas, pesos)    
    particula[1:] = prototipos
    return [particula, pesos]

def testar(teste, classesTeste, particula, classesParticula, pesos):
    n = shape(teste)[0]
    
    distancias = calcularDistancias(particula[1:], teste, pesos, particula[0,:pa], n, k)
    graus = fcm.calcularGraus(distancias)
    particao = argmax(graus,1)       
    classesResultantes = array([classesParticula[prot] for prot in particao])
    numeroErros = float(size(classesTeste[classesTeste != classesResultantes]))
    return (numeroErros / n)*100.0

def rodarValidacaoCruzada(dados, classes, nProts, montecarlo, nFolds, np): 
    global pa
    global k
    global n    
    global treinamento
    global classesTreinamento
    
    if len(nProts) == 1:
        nProts = ones(max(classes)+1) * nProts[0] 
    
    k = sum(nProts)
    
    erros = zeros(montecarlo*nFolds)
    pa = shape(dados)[1]/2
    if not intervalar:
        pa = shape(dados)[1]
    consideradas = zeros(pa)
    nDados = size(classes)
    for i in range(montecarlo):
        indices = arange(nDados)
        random.shuffle(indices)
        dadosEmbaralhados = dados[indices,:]
        classesEmbaralhadas = classes[indices]
        folds = slvq.separarFolds(dadosEmbaralhados, classesEmbaralhadas, nFolds)
        for fold in range(nFolds):
            print i*nFolds + fold
            [treinamento, classesTreinamento, teste, classesTeste] = slvq.separarConjuntos(folds, dadosEmbaralhados, classesEmbaralhadas, fold)      
            n = shape(treinamento)[0]
            inicializar(nProts, np)
            treinar(np, nProts)
            #print GBEST
            consideradas = consideradas + GBEST[0,:pa]
            erros[i*nFolds + fold] =  testar(teste, classesTeste, GBEST, classesGBEST, pesosGBEST)
            print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo * nFolds)
    
def lerDados(nome):
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
    
    global intervalar
    global alfa
    global beta
    
    intervalar = True
    alfa = 0.4
    beta = 1.0 - alfa
    
    prots = [2,2]
    nome = "cogumelos2Classes.txt"
    
    print "pso", alfa, beta, "velocidadeajustadacorrigidanovocriterio2selecao", prots
    print nome
    [dados, classes] = lerDados(nome)
    rodarValidacaoCruzada(dados, classes, prots, 10, 10, 20)      