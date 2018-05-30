from numpy import *
from scipy.spatial.distance import cdist

import fastslvq as slvq
import fcm as fcm


def distanciaVetorizavelClassica(indice,pesos,prots,consideradas):
    distancias[:, indice] = cdist([consideradas*sqrt(pesos[indice])*prots[indice]],consideradas*sqrt(pesos[indice])*dados)[0]**2

def distanciaVetorizavelIntervalar(indice,pesos,prots,consideradas):
    pMins = cdist([consideradas*sqrt(pesos[indice])*prots[indice,::2]],consideradas*sqrt(pesos[indice])*dados[:,::2])[0]
    pMaxs = cdist([consideradas*sqrt(pesos[indice])*prots[indice,1::2]],consideradas*sqrt(pesos[indice])*dados[:,1::2])[0]
    distancias[:, indice] = (pMins**2 + pMaxs**2)/sum(consideradas)

def calcularDistancias(prototipos, dadosp, pesos, variaveisConsideradas, n, k, removidosC):
    global distancias   
    global dados
    
    dados = dadosp    
    consideradas = around(variaveisConsideradas) 
    distancias = zeros((n,k))
    
    if intervalar:
        distanciaIntervalarVetorizada(arange(k)[logical_not(removidosC)],pesos,prototipos,consideradas)
    else:
        distanciaClassicaVetorizada(arange(k)[logical_not(removidosC)],pesos,prototipos,consideradas)
    return distancias

def gerarSolucao(nProts):
    variaveisConsideradas = random.rand(shape(treinamento)[1])    
    [prototipos,classes] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts)
        
    fonte = ajustarMinsMaxs(ajustarVariaveisConsideradas(append([variaveisConsideradas],prototipos,axis=0)))    
    [pesos,fonte[1:],removidos] = calcularPesosPrototipos(fonte[1:], classes, fonte[0,:pa])
    return [fonte, classes, pesos, removidos]

def iniciarVelocidades(np):
    global velocidades
    velocidades = random.uniform(-1,1,(np,k+1,(intervalar+1)*pa))

def iniciarVelocidade():
    velocidade = random.uniform(-1,1,(k+1,(intervalar+1)*pa))
    return velocidade

def ajustarVelocidade(velocidade): 
    velocidade[velocidade > 1] = 1
    velocidade[velocidade < -1] = -1
    return velocidade

def inicializar(nProts, np):
    global particulas
    global pesosParticulas
    global classesParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global indiceGBEST
    global mins
    global maxs
    global criterios
    global limites
    global removidos
    global removidosPBEST
    
    limites = zeros(np)
    removidos = zeros((np, k), dtype=bool)
    
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
        [particulas[particula], classesParticulas[particula], pesosParticulas[particula], removidos[particula]] = gerarSolucao(nProts)
        criterios[particula] = calcularCriterioJ(particulas[particula], pesosParticulas[particula], classesParticulas[particula], removidos[particula])
    PBEST = copy(particulas)
    criteriosPBEST = copy(criterios)
    pesosPBEST = copy(pesosParticulas)
    removidosPBEST = copy(removidos)
    indiceGBEST = argmin(criterios) 
 
def calcularCriterioJ(particula, pesos, classesParticula, removidosJ):
    [erro,distancias] = testar(treinamento, classesTreinamento, particula, classesParticula, pesos, removidosJ)
    
    sumDistanciasProtsMembros = sum(distancias.min(1) * (classesTreinamento == classesParticula[argmin(distancias,1)]))
    criterio = sumDistanciasProtsMembros / len(classesTreinamento)
    if criterio > 1 or isnan(criterio):
        print "opa", criterio
    return alfa * (erro/100) + beta * criterio            

def ajustarVariaveisConsideradas(particula):
    consideradas = particula[0,:pa]
    consideradas[consideradas < 0] = 0
    consideradas[consideradas > 1] = 1
    while all(around(consideradas) == 0):
        consideradas = random.rand(len(consideradas))
    particula[0,:pa] = consideradas
    return particula

def ajustarMinsMaxs(dados):
    temp = dados[1:]
    temp[temp < 0] = 0
    temp[temp > 1] = 1 
    dados[1:] = temp
    if intervalar:
        minimos = dados[1:,::2]
        maximos = dados[1:,1::2]
        ind = minimos > maximos
        if any(ind == True):
            temp = minimos[ind]
            minimos[ind] = maximos[ind]
            maximos[ind] = temp
            
            dados[1:,::2] = minimos
            dados[1:,1::2] = maximos   
    return dados

def calcularPesosPrototipos(prototipos, classesPrototipos, variaveisConsideradas, removidosC = [], pesos = []):
    nClasses = max(classesTreinamento + 1)
    consideradas = around(variaveisConsideradas)
    if len(removidosC) == 0:
        removidosC = zeros(k, dtype=bool)
    if len(pesos) == 0:
        pesos = ones((k,pa))
    distancias = calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k, removidosC)
    distancias[:,removidosC] = float('Inf')
    particao = argmin(distancias,1)
    deltas = zeros((k,pa))
    for prot in range(k):
        if removidosC[prot] == False:
            membros = fcm.getMembros(treinamento, classesTreinamento, particao, classesPrototipos, prot)
            if size(membros) > 0:
                if intervalar:
                    mi = ((prototipos[prot,::2] - membros[:,::2])**2)
                    ma = ((prototipos[prot,1::2] - membros[:,1::2])**2)
                    diff = (mi + ma)/sum(consideradas)
                else:
                    diff = ((prototipos[prot] - membros)**2)/sum(consideradas)
                        
                deltas[prot,:] = sum(consideradas * diff,0)
                distancias[where((particao==prot).__and__(classesTreinamento != classesPrototipos[prot]))[0],:] = 0
        distancias[where(particao!=prot)[0],prot] = 0
    somatoriosClasses = zeros(nClasses)
    for classe in range(nClasses):
        somatoriosClasses[classe] = sum(distancias[:,classesPrototipos == classe])
    if any(somatoriosClasses == 0):
        pesosClasses = ones(nClasses)
    else:
        pesosClasses = (prod(somatoriosClasses)**(1.0/nClasses))/somatoriosClasses
    pesosPrototipos = ones(k)
    for classe in range(nClasses):
        if somatoriosClasses[classe] > 0:
            protsClasse = where(classesPrototipos == classe)[0]
            somatorioPrototipos = sum(distancias[:,protsClasse],axis=0)
            if any(somatorioPrototipos == 0):
                removidosC[protsClasse[somatorioPrototipos == 0]] = True
                prototipos[protsClasse[somatorioPrototipos == 0]] = 0.0
            somatoriosCorretos = somatorioPrototipos > 0
            pesosPrototipos[protsClasse[somatoriosCorretos]] = ((pesosClasses[classe] * prod(somatorioPrototipos[somatoriosCorretos]))**(1.0/sum(somatoriosCorretos)))/somatorioPrototipos[somatoriosCorretos]
            for prot in protsClasse[somatoriosCorretos]:
                delta = deltas[prot,:]
                achou = any((consideradas == 1.0).__and__(delta == 0.0))
                if achou == False:
                    produtorio = (pesosPrototipos[prot]*prod(delta[delta > 0]))**(1.0/sum(consideradas))
                    pesos[prot,:] = array([produtorio / v if v > 0 else 0 for v in delta])
                else:
                    pesos[prot,:] = array([pesosPrototipos[prot]**(1.0/sum(consideradas)) if v > 0 else 0 for v in consideradas])
    return [pesos,prototipos,removidosC]

def VABC(np, nProts):    
    MAX_ITERACAO = 200 
    
    r = 1  
    repeticoes = 0
    f_antes = criteriosPBEST[indiceGBEST]
    while r < MAX_ITERACAO and repeticoes < 50:
        r = r + 1
        # atualizar PBESTs e GBEST
        enviarTrabalhadoras(np)
        enviarObservadoras(np,r,MAX_ITERACAO)
        enviarEscoteiras(np, nProts)
        
        f_depois = criteriosPBEST[indiceGBEST]
        if f_depois == f_antes:
            repeticoes = repeticoes + 1
        else:
            repeticoes = 0      
        f_antes = f_depois 
    
def setarVariavelFonte(fonte,fonteK,prot,j):
    fi = 2 * random.random_sample() -1
    xTemp = copy(particulas[fonte])
    if prot == 0:
        var = particulas[fonte,prot,j] + fi * (particulas[fonte,prot,j] - particulas[fonteK,prot,j])
        if var > 1:
            var = 1
        elif var < 0:
            var = 0
        xTemp[prot,j] = var
        xTemp = ajustarVariaveisConsideradas(xTemp)
    else:
        if intervalar:
            mi = particulas[fonte,prot,2*j] + fi * (particulas[fonte,prot,2*j] - particulas[fonteK,prot,2*j])
            ma = particulas[fonte,prot,2*j+1] + fi * (particulas[fonte,prot,2*j+1] - particulas[fonteK,prot,2*j+1])
            if mi > ma:
                temp = ma
                ma = mi
                mi = temp
            if ma > maxs[j]:
                ma = maxs[j]
            elif ma < mins[j]:
                ma = mins[j]
            if mi > maxs[j]:
                mi = maxs[j]
            elif mi < mins[j]:
                mi = mins[j]
            xTemp[prot,2*j] = mi
            xTemp[prot,2*j+1] = ma
        else:
            var = particulas[fonte,prot,j] + fi * (particulas[fonte,prot,j] - particulas[fonteK,prot,j])
            if var > maxs[j]:
                var = maxs[j]
            elif var < mins[j]:
                var = mins[j]
            xTemp[prot,j] = var
    return xTemp

def enviarTrabalhadoras(np):
    global limites
    global particulas
    global pesosParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global indiceGBEST
    global criterios
    global removidos
    global removidosPBEST
    
    for fonte in range(np):
        j = random.choice(pa) #selecionar uma dimensao aleatoria do problema
        prot = random.choice(append(delete(arange(k),where(removidos[fonte]==True))+1,0)) #selecionar um prototipo aleatorio do problema
        #if prot > 0:
        #    j = random.choice(delete(arange(pa),where(around(particulas[fonte,0,:pa]) ==0.0)))
        fonteK = random.choice(delete(arange(np),fonte)) #selecionar uma fonte aleatoria
        
        xTemp = setarVariavelFonte(fonte,fonteK,prot,j)
        pe = pesosParticulas[fonte]
        if prot == 0:
            pe = []
        [pesos, xTemp[1:], rTemp] = calcularPesosPrototipos(xTemp[1:], classesParticulas[fonte], xTemp[0,:pa], removidos[fonte], pe)        
        f = calcularCriterioJ(xTemp, pesos, classesParticulas[fonte], rTemp)
        if f < criterios[fonte]:
            criterios[fonte] = f
            particulas[fonte] = xTemp
            pesosParticulas[fonte] = pesos
            removidos[fonte] = rTemp
            limites[fonte] = 0
        else:
            limites[fonte] = limites[fonte] + 1
        if f < criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = f
            PBEST[fonte] = xTemp
            pesosPBEST[fonte] = pesos  
            removidosPBEST[fonte] = rTemp    
        
    indiceGBEST = argmin(criteriosPBEST)

def enviarObservadoras(np, t, tmax):
    global limites
    global particulas
    global velocidades
    global pesosParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global indiceGBEST
    global criterios
    global removidos
    global removidosPBEST
    
    fitness = 1.0 / (criterios + 1.0)
    probs = fitness / sum(fitness)
    
    wmin = 0.4
    wmax = 0.9
    
    w = wmin+(wmax-wmin)*(tmax-t)/tmax
    c1 = 0.5+ random.rand()/2.0
    c2 = 0.5+ random.rand()/2.0

    for index in range(np):    
        fonte = random.choice(int(np),p=probs)
        r1 = random.rand()
        r2 = random.rand()
        
        velocidades[fonte] = ajustarVelocidade(w*velocidades[fonte] + c1*r1*(PBEST[fonte] - particulas[fonte]) + c2*r2*(PBEST[indiceGBEST] - particulas[fonte]))
          
        consideradasAntes = copy(particulas[fonte,0,:pa])        
        xTemp = ajustarMinsMaxs(ajustarVariaveisConsideradas(particulas[fonte] + velocidades[fonte])) # calculo da nova posicao
        # ajuste da posicao para que os valores fiquem no intervalo [1;c], que sao os possiveis clusters
        pe = pesosParticulas[fonte]
        if any(around(consideradasAntes) != around(xTemp[0,:pa])):
            pe = []
        [pesos, xTemp[1:], rTemp] = calcularPesosPrototipos(xTemp[1:], classesParticulas[fonte], xTemp[0,:pa], removidos[fonte], pe)        
        f = calcularCriterioJ(xTemp, pesos, classesParticulas[fonte], rTemp)
        if f < criterios[fonte]:
            criterios[fonte] = f
            particulas[fonte] = xTemp
            pesosParticulas[fonte] = pesos
            removidos[fonte] = rTemp
            limites[fonte] = 0
        else:
            limites[fonte] = limites[fonte] + 1
        if f < criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = f
            PBEST[fonte] = xTemp
            pesosPBEST[fonte] = pesos  
            removidosPBEST[fonte] = rTemp     
        if f < criteriosPBEST[indiceGBEST]:
            indiceGBEST = fonte

def refinar(indice):
    t = 1
    tMax = 5
    particula = copy(particulas[indice])
    prototipos = copy(particula[1:])
    variaveisConsideradas = particula[0,:pa]
    pesos = copy(pesosParticulas[indice])
    removidosR = copy(removidos[indice])
    classesParticula = classesParticulas[indice]
    
    epsilon = 0.00001
    Jatual = 1
    Jdepois = -1
    while t <= tMax and abs(Jatual - Jdepois)>epsilon:
        #fase dos prototipos
        distancias = calcularDistancias(prototipos, treinamento, pesos, variaveisConsideradas, n, k, removidosR)
        distancias[:,removidosR] = float('Inf')
                
        Jatual = Jdepois
        particao = argmin(distancias,1)
        Jdepois = sum((distancias.min(1) / sum(around(particula[0,:pa]))) * (classesTreinamento == classesParticula[particao])) / len(classesTreinamento)
        
        for prot in arange(k)[logical_not(removidosR)]:
            membros = (particao == prot).__and__(classesTreinamento == classesParticula[prot])
            if any(membros):
                prototipos[prot] = sum(treinamento[membros],0)/sum(membros)
            
        [pesos, prototipos, removidosR] = calcularPesosPrototipos(prototipos, classesParticula, variaveisConsideradas, removidosR, pesos)
        t = t + 1   
    particula[1:] = prototipos
    return [particula, pesos, removidosR]

def enviarEscoteiras(np, nProts):
    global limites
    global particulas
    global velocidades
    global pesosParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global indiceGBEST
    global criterios
    global removidos
    global removidosPBEST
    
    for fonte in arange(np)[limites >= 10]:
        #[xTemp, pesos, rTemp] = refinar(fonte)
        #f = calcularCriterioJ(xTemp, pesos, classesParticulas[fonte], rTemp)
        #if f < criterios[fonte]:
        #    criterios[fonte] = f
        #    particulas[fonte] = xTemp
        #    pesosParticulas[fonte] = pesos
        #    removidos[fonte] = rTemp
        #else:
        [particulas[fonte], classesParticulas[fonte], pesosParticulas[fonte], removidos[fonte]] = gerarSolucao(nProts)
        criterios[fonte] = calcularCriterioJ(particulas[fonte], pesosParticulas[fonte], classesParticulas[fonte], removidos[fonte])
        
        velocidades[fonte] = iniciarVelocidade()
            
        if criterios[fonte] < criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = criterios[fonte]
            PBEST[fonte] = copy(particulas[fonte])
            pesosPBEST[fonte] = copy(pesosParticulas[fonte])  
            removidosPBEST[fonte] = copy(removidos[fonte])  
        limites[fonte] = 0          
            
    indiceGBEST = argmin(criteriosPBEST)

def testar(teste, classesTeste, particula, classesParticula, pesos, removidosT):
    ne = shape(teste)[0]
    
    distancias = calcularDistancias(particula[1:], teste, pesos, particula[0,:pa], ne, k, removidosT)
    distancias[:,removidosT] = float('Inf')
    particao = argmin(distancias,1)       
    classesResultantes = classesParticula[particao]
    numeroErros = float(sum(classesTeste != classesResultantes))
    return [(numeroErros / ne)*100.0, distancias]

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
    rem = 0.0
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
            VABC(np, nProts)
            GBEST = PBEST[indiceGBEST]
            consideradas = consideradas + around(GBEST[0,:pa]).astype(float)
            #print sum(removidosPBEST[indiceGBEST])    
            rem = rem + float(sum(removidosPBEST[indiceGBEST]))           
            [erros[i*nFolds + fold], d] =  testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
            print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo * nFolds)
    print rem / (montecarlo * nFolds)

def rodarLOO(dados, classes, nProts, montecarlo, np): 
    global pa
    global k
    global n    
    global treinamento
    global classesTreinamento
    
    if len(nProts) == 1:
        nProts = ones(max(classes)+1) * nProts[0] 
    
    k = sum(nProts)
    rem = 0.0
    pa = shape(dados)[1]/2
    if not intervalar:
        pa = shape(dados)[1]
    consideradas = zeros(pa)
    nDados = size(classes)
    erros = zeros(montecarlo*nDados)
    for i in range(montecarlo):
        indices = arange(nDados)
        random.shuffle(indices)
        dadosEmbaralhados = dados[indices,:]
        classesEmbaralhadas = classes[indices]
        for dado in range(nDados):
            print i*nDados + dado
            teste = array([dadosEmbaralhados[dado,:]])
            classesTeste = array([classesEmbaralhadas[dado]])
            indTreinamento = delete(arange(nDados),dado)
            treinamento = dadosEmbaralhados[indTreinamento,:]
            classesTreinamento = classesEmbaralhadas[indTreinamento]
            n = shape(treinamento)[0]
            inicializar(nProts, np)
            VABC(np, nProts)
            GBEST = PBEST[indiceGBEST]
            consideradas = consideradas + GBEST[0,:pa]
            print sum(removidosPBEST[indiceGBEST])    
            rem = rem + float(sum(removidosPBEST[indiceGBEST]))        
            [erros[i*nDados + dado], d] =  testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
            print erros[i*nDados + dado]
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo * nDados)
    print rem / (montecarlo * nDados)

def geraNormaisMultiVariadas(parametros,intervalo):
    quantidades = parametros[:,-2]
    nDados = sum(quantidades)
    dados = zeros((nDados,2*pa))
    classes = zeros(nDados)
    nRegioes = shape(parametros)[0]
    inicio = 0
    fim = 0
    par = parametros[:,0:pa*2]
    for regiao in range(nRegioes):
        inicio = fim
        fim = fim + quantidades[regiao]
        medias = par[regiao,0::2]#array([parametros[regiao,0], parametros[regiao,2]])
        covs = diag(par[regiao,1::2])#array([parametros[regiao,1], parametros[regiao,3]]))
        d = random.multivariate_normal(medias,covs,quantidades[regiao])
        delta = random.randint(1,intervalo,pa)
        delta = delta.astype(float)
        for var in range(pa):
            dados[inicio:fim,2*var] = d[:,var] - (delta[var]/2)
            dados[inicio:fim,2*var+1] = d[:,var] + (delta[var]/2)
        classes[inicio:fim] = parametros[regiao,-1]
    return [dados,classes.astype(int)]
    
def rodarSimulados(parametros, intervalo, nProts, montecarlo, np): 
    global pa
    global k
    global n    
    global treinamento
    global classesTreinamento
    
    if len(nProts) == 1:
        nProts = ones(max(parametros[:,5])+1) * nProts[0] 
    
    k = sum(nProts)
    
    erros = zeros(montecarlo)
    rem = 0.0
    pa = (shape(parametros)[1] - 2)/2
    consideradas = zeros(pa)
    for i in range(montecarlo):
        print i
        [dados,classes] = geraNormaisMultiVariadas(parametros,intervalo)
        
        #qtd = shape(dados)[0]
        #for instancia in range(qtd):
        #    print dados[instancia,], classes[instancia]
        #exit()
        for var in arange(pa):
            dados[:,[2*var,2*var+1]] = (dados[:,[2*var,2*var+1]] - dados[:,[2*var,2*var+1]].min())/(dados[:,[2*var,2*var+1]].max() - dados[:,[2*var,2*var+1]].min())
        
        [treinamento, classesTreinamento, teste, classesTeste] = slvq.separarHoldOut(dados, classes, 0.5)
        n = shape(treinamento)[0]
        inicializar(nProts, np)
        VABC(np, nProts)
        GBEST = PBEST[indiceGBEST]
        #print GBEST[0,:pa]
        consideradas = consideradas + around(GBEST[0,:pa]).astype(float)
        #print sum(removidosPBEST[indiceGBEST])    
        rem = rem + float(sum(removidosPBEST[indiceGBEST]))          
        [erros[i], d] =  testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
        print erros[i]
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo)
    print rem / (montecarlo)    

def rodarTrainTest(dadosTrain, classesTrain, dadosTest, classesTest, nProts, montecarlo, np):
    global pa
    global k
    global n
    global treinamento
    global classesTreinamento

    k = sum(nProts)

    erros = zeros(montecarlo)
    pa = shape(dadosTrain)[1]/2
    rem = 0.0
    consideradas = zeros(pa)
    treinamento = dadosTrain
    classesTreinamento = classesTrain
    teste = dadosTest
    classesTeste = classesTest
    for i in range(montecarlo):
        print i

        n = shape(treinamento)[0]
        inicializar(nProts, np)
        VABC(np, nProts)
        GBEST = PBEST[indiceGBEST]
        consideradas = consideradas + around(GBEST[0,:pa]).astype(float)
        #print sum(removidosPBEST[indiceGBEST])
        rem = rem + float(sum(removidosPBEST[indiceGBEST]))
        [erros[i], d] =  testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
        print erros[i]
    print erros
    print mean(erros)
    print std(erros)
    print consideradas / (montecarlo * nFolds)
    print rem / (montecarlo * nFolds)
    return erros , "\n erro medio:" , mean(erros) , "\n desvio:" , std(erros)

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
    global distanciaClassicaVetorizada
    global distanciaIntervalarVetorizada
    
    distanciaClassicaVetorizada = vectorize(distanciaVetorizavelClassica, excluded=[1,2,3])
    distanciaIntervalarVetorizada = vectorize(distanciaVetorizavelIntervalar, excluded=[1,2,3])
    
    intervalar = True
    alfa = 0.4
    beta = 1.0 - alfa
    
    prots = [10,10,10,10,10,10,10,10,10,10]
    # nome = "covtype_intervals.txt"
    nomeTrain = "Train_Arabic_Digit_intervals_normalizados.txt"
    nomeTest = "Test_Arabic_Digit_intervals_normalizados.txt"
    
    print "vabc", alfa, beta, "tradicionalhardotimizado/100", prots, "tresniveis"
    print nomeTrain
    [dadosTrain, classesTrain] = lerDados(nomeTrain)
    [dadosTest, classesTest] = lerDados(nomeTest)
    # [dados, classes] = lerDados(nome)
    #rodarValidacaoCruzada(dados, classes, prots, 10, 10, 25)
    rodarTrainTest(dadosTrain, classesTrain, dadosTest, classesTest, prots, 30, 25)
    #rodarLOO(dados, classes, prots, 10, 25) 
    #for intervalo in [10]:
    #    print "intervalo", intervalo
    #    rodarSimulados(array([[99,9,99,169,200,0],[108,9,99,169,200,1]] ), intervalo, prots, 100, 25)