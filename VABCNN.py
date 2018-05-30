from numpy import *

import old.fastslvq as slvq
import old.fcm as fcm
from scipy.spatial.distance import cdist
import sys
from old.util import sendMail
from util.functions import print_confusion_matrix

def distanciaVetorizavelClassica(indice,pesos,prots,consideradas):
    distancias[:, indice] = cdist([consideradas*sqrt(pesos[indice])*prots[indice]],consideradas*sqrt(pesos[indice])*dados,'sqeuclidean')[0]

def distanciaVetorizavelIntervalar(indice,pesos,prots,consideradas):
    pMins = cdist([consideradas*sqrt(pesos[indice])*(prots[indice,::2]/(maxs-mins))],consideradas*sqrt(pesos[indice])*(dados[:,::2]/(maxs-mins)),'sqeuclidean')[0]
    pMaxs = cdist([consideradas*sqrt(pesos[indice])*(prots[indice,1::2]/(maxs-mins))],consideradas*sqrt(pesos[indice])*(dados[:,1::2]/(maxs-mins)),'sqeuclidean')[0]
    distancias[:, indice] = (pMins + pMaxs)/(2*sum(consideradas))

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
    #velocidades[:,1:,::2] = velocidades[:,1:,::2]*(maxs-mins)
    #velocidades[:,1:,1::2] = velocidades[:,1:,1::2]*(maxs-mins)

def iniciarVelocidade():
    velocidade = random.uniform(-1,1,(k+1,(intervalar+1)*pa))
    #velocidade[1:,::2] = velocidade[1:,::2]*(maxs-mins)
    #velocidade[1:,1::2] = velocidade[1:,1::2]*(maxs-mins)
    return velocidade

def ajustarVelocidade(velocidade): 
    velocidade[velocidade > 1] = 1
    velocidade[velocidade < -1] = -1
    
    #minimos = velocidade[1:,::2]
    #maximos = velocidade[1:,1::2]
    #indMins = where(minimos < -(maxs-mins))
    #minimos[indMins] = -(maxs-mins)[indMins[1]]
    #indMaxs = where(minimos > (maxs-mins))
    #minimos[indMaxs] = (maxs-mins)[indMaxs[1]]
    
    #indMins = where(maximos < -(maxs-mins))
    #maximos[indMins] = -(maxs-mins)[indMins[1]]
    #indMaxs = where(maximos > (maxs-mins))
    #maximos[indMaxs] = (maxs-mins)[indMaxs[1]]
    #velocidade[1:,::2] = minimos
    #velocidade[1:,1::2] = maximos
    return velocidade

def inicializar(nProts, np):
    global particulas
    global pesosParticulas
    global classesParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global indiceGBEST
    global criterios
    global limites
    global removidos
    global removidosPBEST
    
    limites = zeros(np)
    removidos = zeros((np, k), dtype=bool)
    
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
    [erro,criterio] = testar(treinamento, classesTreinamento, particula, classesParticula, pesos, removidosJ)
    #sumDistanciasProtsMembros = sum(distancias.min(1) * (classesTreinamento == classesParticula[argmin(distancias,1)]))
    #criterio = sumDistanciasProtsMembros / len(classesTreinamento)
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
    #temp = dados[1:]
    #temp[temp < 0] = 0
    #temp[temp > 1] = 1 
    #dados[1:] = temp
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
        indMins = where(minimos < mins)
        minimos[indMins] = mins[indMins[1]]
        indMaxs = where(minimos > maxs)
        minimos[indMaxs] = maxs[indMaxs[1]]
        indMins = where(maximos < mins)
        maximos[indMins] = mins[indMins[1]]
        indMaxs = where(maximos > maxs)
        maximos[indMaxs] = maxs[indMaxs[1]]
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
    kVizinhos = argsort(distancias,1)[:,:min(KNN,k-sum(removidosC))]
    kDists = distancias[arange(n)[:,None], kVizinhos]
    classesCertas = (classesPrototipos[kVizinhos] == classesTreinamento[:,None])
    kDistsCertas = classesCertas.astype(float) * kDists
    
    qntds = sum(classesCertas,1)
    ne2 = sum(qntds > 0)
    qntds[qntds == 0] = 1
    
    distsPrototipos = array([sum(kDistsCertas[kVizinhos == prot]) for prot in arange(k)])
    temp = ((2*classesCertas -1 )*(kVizinhos+1)).ravel()
    ocorrencias = bincount(temp[temp>=0],minlength=k+1)[1:]
    classInfo = array([[sum(distsPrototipos[classesPrototipos == classe]),sum((distsPrototipos > 0).__and__(classesPrototipos == classe)), prod(distsPrototipos[(distsPrototipos > 0).__and__(classesPrototipos == classe)]/ocorrencias[(distsPrototipos > 0).__and__(classesPrototipos == classe)])] for classe in arange(nClasses)])
    
    sumClasses = classInfo[:,0]/classInfo[:,1]
    sumClasses[isnan(sumClasses)] = 0.0
    if any(sumClasses == 0):
        pesosClasses = ones(nClasses)
        for classe in arange(nClasses):
            if sumClasses[classe] == 0.0:
                membrosClasse = classesPrototipos == classe
                [prototipos[membrosClasse,:], classesPrototipos[membrosClasse]] = slvq.iniciarPrototiposPorSelecaoPorClasse(treinamento, classesTreinamento, sum(membrosClasse), classe)
                removidosC[membrosClasse] = False
                pesos[membrosClasse,:] = 1.0
    else:
        pesosClasses = (prod(sumClasses)**(1.0/nClasses))/sumClasses        
    
    pesosPrototipos = ones(k)    
    nPrototipos = classInfo[:,1]
    indProts = arange(k)[logical_not(removidosC)].astype(int)
    prodsClasses = classInfo[:,2]
    for prot in indProts:
        classe = int(classesPrototipos[prot])
        if sumClasses[classe] > 0.0:
            if distsPrototipos[prot] == 0:
                    removidosC[prot] = True
            else:
                pesosPrototipos[prot] = ((pesosClasses[classe] * prodsClasses[classe])**(1.0/classInfo[classe,1]))/(distsPrototipos[prot]/ocorrencias[prot])
                membros = where((kVizinhos==prot).__and__(classesCertas))[0]
                if intervalar:
                    mi = (((prototipos[prot,::2] - treinamento[membros,::2])/(maxs-mins))**2)
                    ma = (((prototipos[prot,1::2] - treinamento[membros,1::2])/(maxs-mins))**2)
                    diff = (mi + ma)/(2*sum(consideradas))
                else:
                    diff = (((prototipos[prot] - treinamento[membros,:])/(maxs-mins))**2)/sum(consideradas)
                delta = sum(consideradas * (diff/qntds[membros,None]/ne2),0)
                achou = any((consideradas == 1.0).__and__(delta == 0.0))
                if achou == False:
                    pesos[prot,:] = array([((pesosPrototipos[prot]*prod(delta[delta > 0]))**(1.0/sum(consideradas)))/v if v > 0 else 1 for v in delta])
                else:
                    pesos[prot,:] = array([pesosPrototipos[prot]**(1.0/sum(consideradas)) if v > 0 else 1 for v in consideradas])
    #if any(isnan(pesos)):
    #    print "isnan"            
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
    numeroErros = 0.0
    numeroClasses = classesParticula.max() + 1
    kVizinhos = argsort(distancias,1)[:,:min(KNN,k-sum(removidosT))]
    kDists = distancias[arange(ne)[:,None], kVizinhos]
    inversas = 1.0/kDists
    membros = array([classesParticula[kVizinhos] == classe for classe in arange(numeroClasses)])
    omegas = sum(membros.astype(float)*inversas,2).T
    #quantidades = sum(membros,2).T
    #fatores = (quantidades.astype(float)/min(KNN,k-sum(removidosT)))*omegas
    #probabilidades = fatores / sum(fatores,1)[:,None]
    vencedoras = argmax(omegas,1)
    indicesSaoZero = where(kDists == 0.0)[0]
    for ind in indicesSaoZero.astype(int):
        kd = kDists[ind,:]
        if sum(kd == 0.0) == 1:
            vencedoras[ind] = classesParticula[kVizinhos[ind,0]]
        else:
            vencedoras[ind] = argmax(bincount(classesParticula[kVizinhos[ind,
                                                                         kd==0]].astype(int),minlength=int(numeroClasses)))
    if len(classesTeste) < 200:
        return vencedoras
    classesCertas = (classesParticula[kVizinhos] == classesTeste[:,None])
    kDistsCertas = classesCertas.astype(float) * kDists
    qntds = sum(classesCertas,1)
    ne2 = sum(qntds > 0)
    qntds[qntds == 0] = 1
    criterio = sum(kDistsCertas / qntds[:,None])/ne2
    if criterio > 1 or isnan(criterio):
        print "opa", criterio
    
    return [(float(sum(vencedoras != classesTeste))/ne)*100.0,criterio]

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
            preds = zeros((30, len(classesTeste)))
            from tqdm import tqdm
            for l in tqdm(arange(30)):
                inicializar(nProts, np)
                VABC(np, nProts)
                GBEST = PBEST[indiceGBEST]
                consideradas = consideradas + around(GBEST[0,:pa]).astype(float)
                #print sum(removidosPBEST[indiceGBEST])
                rem = rem + float(sum(removidosPBEST[indiceGBEST]))
                preds[l] = testar(teste, classesTeste, GBEST,
                                  classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
            # [erros[i*nFolds + fold], d] = testar(teste, classesTeste, GBEST, classesParticulas[indiceGBEST], pesosPBEST[indiceGBEST], removidosPBEST[indiceGBEST])
            # print erros[i*nFolds + fold]
            predictions = around(mean(preds, axis=0))
            print_confusion_matrix(classesTeste, predictions)
            exit()
    print erros
    print mean(erros)
    print std(erros)        
    print consideradas / (montecarlo * nFolds)
    print rem / (montecarlo * nFolds)
    return erros , "\n erro medio:" , mean(erros) , "\n desvio:" , std(erros) , "\n" , consideradas / (montecarlo * nFolds) , "\n" , rem / (montecarlo * nFolds)

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
        covs = diag(par[regiao,1::2]).astype(float)#array([parametros[regiao,1], parametros[regiao,3]]))
        # covs[covs == 0] = 0.75
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
    global mins
    global maxs
    
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
        if intervalar:
            mins = dados[:,::2].min(0)
            maxs = dados[:,1::2].max(0)
        else: 
            mins = dados.min(0)
            maxs = dados.max(0)
        qtd = shape(dados)[0]
        # for instancia in range(qtd):
        #    print dados[instancia,], classes[instancia]
        # exit()
        #for var in arange(pa):
        #    dados[:,[2*var,2*var+1]] = (dados[:,[2*var,2*var+1]] - dados[:,[2*var,2*var+1]].min())/(dados[:,[2*var,2*var+1]].max() - dados[:,[2*var,2*var+1]].min())
        
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
    
    return erros , "\n erro medio:" , mean(erros) , "\n desvio:" , std(erros) , "\n" , consideradas / (montecarlo) , "\n" , rem / (montecarlo)


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
    return erros , "\n erro medio:" , mean(erros) , "\n desvio:" , std(erros) , "\n" , consideradas / (montecarlo * nFolds) , "\n" , rem / (montecarlo * nFolds)


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
    global KNN
    global mins
    global maxs
    
    distanciaClassicaVetorizada = vectorize(distanciaVetorizavelClassica, excluded=[1,2,3])
    distanciaIntervalarVetorizada = vectorize(distanciaVetorizavelIntervalar, excluded=[1,2,3])
    
    intervalar = True
    alfa = 1.0
    beta = 1.0 - alfa
    # 935  763 1734  600 1598
    prots = [34, 28, 12, 20, 12, 42, 38, 34, 6, 40]
    nome = "base_meses_corrigida_nao_normalizados.txt"
    
    KNN = 8
    
    # titulo = "vabc", alfa, beta, KNN, "tradicionalhardotimizado/100", prots, "tresniveis", "2,2 novo"
    # print titulo
    # print nome
    # titulo = titulo , nome
    [dados, classes] = lerDados(nome)

    # nomeTrain = "Train_Arabic_Digit_intervals_normalizados.txt"
    # nomeTest = "Test_Arabic_Digit_intervals_normalizados.txt"
    # titulo = titulo , nomeTrain
    # [dadosTrain, classesTrain] = lerDados(nomeTrain)
    # [dadosTest, classesTest] = lerDados(nomeTest)

    if intervalar:
        mins = dados[:,::2].min(0)
        maxs = dados[:,1::2].max(0)
    else:
        mins = dados.min(0)
        maxs = dados.max(0)
    #europeus [20,40] k = 3
    #secos [28,38,34] k = 21
    #todos [34, 28, 12, 20, 12, 42, 38, 34, 6, 40] k = 8
    texto = rodarValidacaoCruzada(dados, classes, prots, 10, 10, 25)
    #rodarLOO(dados, classes, prots, 10, 25)
    # texto = rodarTrainTest(dadosTrain, classesTrain, dadosTest, classesTest, prots, 30, 25)
    # intervalo = 10
    #2,1 [[99,9,99,169,200,0],[108,9,99,169,200,1]] k = 3
    #2,2 [[99,9,99,169,200,0],[104,16,138,16,150,0],[104,16,60,16,150,1],[108,9,99,169,200,1]] k = 21
    #3,2 [[99,9,99,169,44,25,200,0],[104,16,138,16,44,25,150,0],[104,16,60,16,41,25,150,1],[108,9,99,169,41,25,200,1]] k = 2
    # texto = rodarSimulados(array([[99,9,99,169,200,0],[104,16,118,16,150,0],[104,16,80,16,150,1],[100,9,99,169,200,1]]), intervalo, prots, 100, 25)
    # sendMail(titulo, texto)
