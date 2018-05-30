from numpy import *

import fastslvq as slvq
import fcm as fcm


def gerarSolucao(nProts):
    [prototipos,classes] = slvq.iniciarPrototiposPorSelecao(treinamento, classesTreinamento, nProts)
            
    pesos = calcularPesosPrototipos(prototipos, classes)
    return [prototipos, classes, pesos]

def iniciarVelocidades(np):
    global velocidades
    if intervalar:
        velocidades = random.uniform(-1,1,(np,k,2*pa))
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
        velocidade = random.uniform(-1,1,(k,2*pa))
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
    if intervalar:
        for var in range(pa):
            minimos = velocidade[:,2*var]
            minimos[minimos < (-1 * (maxs[var]-mins[var]))] = (-1 * (maxs[var]-mins[var]))
            minimos[minimos > (maxs[var]-mins[var])] = (maxs[var]-mins[var])  
            velocidade[:,2*var] = minimos
            maximos = velocidade[:,2*var+1]
            maximos[maximos < (-1 * (maxs[var]-mins[var]))] = (-1 * (maxs[var]-mins[var]))
            maximos[maximos > (maxs[var]-mins[var])] = (maxs[var]-mins[var])   
            velocidade[:,2*var+1] = maximos         
    else:
        for var in range(pa):
            temp = velocidade[:,var]
            temp[temp < (-1 * (maxs[var]-mins[var]))] = (-1 * (maxs[var]-mins[var]))
            temp[temp > (maxs[var]-mins[var])] = (maxs[var]-mins[var]) 
            velocidade[:,var] = temp
    return velocidade

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
    global limites
    
    limites = zeros(np)
    
    if intervalar:
        mins = treinamento[:,::2].min(0)
        maxs = treinamento[:,1::2].max(0)
    else:
        mins = treinamento.min(0)
        maxs = treinamento.max(0)
    particulas = zeros((np,sum(nProts),shape(treinamento)[1]))
    pesosParticulas = ones((np,sum(nProts),pa))
    iniciarVelocidades(np)
    classesParticulas = zeros((np,sum(nProts)))
    criterios = zeros(np)
    for particula in range(np):
        [particulas[particula], classesParticulas[particula], pesosParticulas[particula]] = gerarSolucao(nProts)
        criterios[particula] = calcularCriterioJ(particulas[particula], pesosParticulas[particula], classesParticulas[particula])
    PBEST = copy(particulas)
    criteriosPBEST = copy(criterios)
    pesosPBEST = copy(pesosParticulas)
    indice = argmax(criterios)
    GBEST = copy(particulas[indice])
    classesGBEST = copy(classesParticulas[indice])
    criterioGBEST = criterios[indice]
    pesosGBEST = copy(pesosParticulas[indice])     


def calcularCriterioJ(particula, pesos, classesParticula):
    
    erro = testar(treinamento, classesTreinamento, particula, classesParticula, pesos)/100
    distancias = calcularDistancias(particula, treinamento, pesos, n, k)
    graus = fcm.calcularGraus(distancias)
    #particao = argmin(distancias,1)
    #classesObtidas = classesParticula[particao]
    #criterio = sum(distancias.min(1) * (classesTreinamento == classesParticula[argmin(distancias,1)]))
    
    sumDistanciasProtsMembros = sum(graus.max(1)**2 * distancias.min(1) * (classesTreinamento == classesParticula[argmax(graus,1)]))
    
    nClasses = max(classesTreinamento)+1
    medias = zeros((nClasses, shape(particula)[1]))
    classesTemp = arange(nClasses)
    #prots = particula[1:]
    for classe in classesTemp:
        #protsClasse = prots[classesParticula==classe]
        medias[classe] = treinamento[classesTreinamento == classe].mean(0)#sum(protsClasse,0)/shape(protsClasse)[0]
    distanciasProtCentros = calcularDistancias(particula, medias, pesos, nClasses, k)
    #distanciasMembrosCentros = calcularDistancias(medias, treinamento, ones((nClasses,pa)), particula[0,:pa], n, nClasses)
    #for classe in classesTemp:
    #    distanciasMembrosCentros[classe!=classesObtidas,classe] = 0.0
    for classe in classesTemp:
        distanciasProtCentros[classe,classesParticula != classe] = 0.0
    #criterio = sum(distanciasMembrosCentros) / sum(distanciasProtCentros)    
    criterio = sumDistanciasProtsMembros / sum(distanciasProtCentros)  
       
    return 1/((alfa * erro + beta * criterio)+1)
           
    
def calcularDistancias(prototipos, dados, pesos, n, k):
    distancias = zeros((n,k))
    for prot in range(k):
        if intervalar:
            mins =  pesos[prot,:] * ((prototipos[prot,::2] - dados[:,::2])**2)
            maxs =  pesos[prot,:] * ((prototipos[prot,1::2] - dados[:,1::2])**2)
            distancias[:,prot] = sqrt(sum(mins,1)) + sqrt(sum(maxs,1)) + 0.0000000001
        else:
            distancias[:,prot] = sqrt(sum((pesos[prot,:] * (prototipos[prot,:] - dados)**2),1)) + 0.0000000001
    return distancias

def ajustarMinsMaxs(dados):
    if intervalar:
        for var in range(pa):
            minimos = dados[:,2*var]
            minimos[minimos < mins[var]] = mins[var]
            minimos[minimos > maxs[var]] = maxs[var]
            maximos = dados[:,2*var+1]
            maximos[maximos < mins[var]] = mins[var]
            maximos[maximos > maxs[var]] = maxs[var] 
            
            ind = minimos > maximos
            temp = minimos[ind]
            minimos[ind] = maximos[ind]
            maximos[ind] = temp
                
            dados[:,2*var] = minimos
            dados[:,2*var+1] = maximos         
    else:
        for var in range(pa):
            temp = dados[:,var]
            temp[temp < mins[var]] = mins[var]
            temp[temp > maxs[var]] = maxs[var] 
            dados[:,var] = temp
    return dados

def calcularPesosPrototipos(prototipos, classesPrototipos, pesos = []):
    nClasses = max(classesTreinamento + 1)
    if len(pesos) == 0:
        pesos = ones((k,pa))
    distancias = calcularDistancias(prototipos, treinamento, pesos, n, k)
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
            deltas[prot,:] = sum(grausMembros.reshape((size(grausMembros),1))  * diff,0)
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
            delta = deltas[prot,:]
            
            produtorio = (pesosPrototipos[prot]*prod(delta[delta > 0]))**(1.0/pa)
            pesos[prot,:] = array([produtorio / v if v > 0 else 0 for v in delta])
    return pesos

def VABC(np, nProts):    
    MAX_ITERACAO = 200 
    
    r = 1  
    repeticoes = 0
    f_antes = criterioGBEST
    while r < MAX_ITERACAO and repeticoes < 50:
        r = r + 1
        # atualizar PBESTs e GBEST
        enviarTrabalhadoras(np)
        enviarObservadoras(np)
        enviarEscoteiras(np, nProts)
        
        f_depois = criterioGBEST
        if f_depois == f_antes:
            repeticoes = repeticoes + 1
        else:
            repeticoes = 0      
        f_antes = f_depois 
    
def setarVariavelFonte(fonte,fonteK,prot,j):
    fi = 2 * random.random_sample() -1
    xTemp = copy(particulas[fonte])
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
    global GBEST
    global classesGBEST
    global criterioGBEST
    global pesosGBEST
    global criterios
    
    for fonte in range(np):
        j = random.choice(pa) #selecionar uma dimensao aleatoria do problema
        prot = random.choice(k) #selecionar um prototipo aleatorio do problema
        fonteK = random.choice(np) #selecionar uma fonte aleatoria
        while fonteK == fonte:
            fonteK = random.choice(np)
        xTemp = setarVariavelFonte(fonte,fonteK,prot,j)
        pe = pesosParticulas[fonte]
        pesos = calcularPesosPrototipos(xTemp, classesParticulas[fonte], pe)        
        f = calcularCriterioJ(xTemp, pesos, classesParticulas[fonte])
        if f > criterios[fonte]:
            criterios[fonte] = f
            particulas[fonte] = copy(xTemp)
            pesosParticulas[fonte] = copy(pesos)
        else:
            limites[fonte] = limites[fonte] + 1
        if f > criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = f
            PBEST[fonte] = copy(xTemp)
            pesosPBEST[fonte] = copy(pesos)        
        if f > criterioGBEST:
            GBEST = copy(xTemp)
            criterioGBEST = f
            classesGBEST = copy(classesParticulas[fonte])
            pesosGBEST = copy(pesos)

def enviarObservadoras(np):
    global limites
    global particulas
    global velocidades
    global pesosParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global GBEST
    global classesGBEST
    global criterioGBEST
    global pesosGBEST
    global criterios
    
    w = 0.6
    c1 = 2
    c2 = 2

    for fonte in range(np):    
        r1 = random.rand()
        r2 = random.rand()
        
        velocidades[fonte] = ajustarVelocidade(w*velocidades[fonte] + c1*r1*(PBEST[fonte] - particulas[fonte]) + c2*r2*(GBEST - particulas[fonte]))
             
        xTemp = ajustarMinsMaxs(particulas[fonte] + velocidades[fonte]) # calculo da nova posicao
        # ajuste da posicao para que os valores fiquem no intervalo [1;c], que sao os possiveis clusters
        pe = pesosParticulas[fonte]
        pesos = calcularPesosPrototipos(xTemp, classesParticulas[fonte], pe)        
        f = calcularCriterioJ(xTemp, pesos, classesParticulas[fonte])
        if f > criterios[fonte]:
            criterios[fonte] = f
            particulas[fonte] = copy(xTemp)
            pesosParticulas[fonte] = copy(pesos)
        else:
            limites[fonte] = limites[fonte] + 1
        if f > criteriosPBEST[fonte]:
            criteriosPBEST[fonte] = f
            PBEST[fonte] = copy(xTemp)
            pesosPBEST[fonte] = copy(pesos)        
        if f > criterioGBEST:
            GBEST = copy(xTemp)
            criterioGBEST = f
            classesGBEST = copy(classesParticulas[fonte])
            pesosGBEST = copy(pesos)

def enviarEscoteiras(np, nProts):
    global limites
    global particulas
    global velocidades
    global pesosParticulas
    global PBEST
    global criteriosPBEST
    global pesosPBEST
    global GBEST
    global classesGBEST
    global criterioGBEST
    global pesosGBEST
    global criterios
    
    for fonte in range(np):
        if limites[fonte] > 10:
            [particulas[fonte], classesParticulas[fonte], pesosParticulas[fonte]] = gerarSolucao(nProts)
            criterios[fonte] = calcularCriterioJ(particulas[fonte], pesosParticulas[fonte], classesParticulas[fonte])
            
            velocidades[fonte] = iniciarVelocidade()
            limites[fonte] = 0
            
            if criterios[fonte] > criteriosPBEST[fonte]:
                criteriosPBEST[fonte] = criterios[fonte]
                PBEST[fonte] = copy(particulas[fonte])
                pesosPBEST[fonte] = copy(pesosParticulas[fonte]) 
            
            if criterios[fonte] > criterioGBEST:
                GBEST = copy(particulas[fonte])
                criterioGBEST = criterios[fonte]
                classesGBEST = copy(classesParticulas[fonte])
                pesosGBEST = copy(pesosParticulas[fonte])

def testar(teste, classesTeste, particula, classesParticula, pesos):
    n = shape(teste)[0]
    
    distancias = calcularDistancias(particula, teste, pesos, n, k)
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
            erros[i*nFolds + fold] =  testar(teste, classesTeste, GBEST, classesGBEST, pesosGBEST)
            print erros[i*nFolds + fold]
    print erros
    print mean(erros)
    print std(erros)        
    
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
    alfa = 0.6
    beta = 1 - alfa
    
    prots = [10,22]
    nome = "mediterraneo_oceanico_normalizados.txt"
    
    print "vabc", alfa, beta, "tradicionalsemconsideradas", prots
    print nome
    [dados, classes] = lerDados(nome)
    rodarValidacaoCruzada(dados, classes, prots, 10, 10, 25)      
