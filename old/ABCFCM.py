from numpy import *

import fastslvq as slvq
import fcm as fcm


def inicializar(treinamento, classesTreinamento, nProts, nAbelhas, intervalar = True):
    np = int(nAbelhas/2.0)
    if intervalar:
        p = shape(treinamento)[1]/2
        mins = treinamento[:,::2].min(0)
        maxs = treinamento[:,1::2].max(0)
    else:
        mins = treinamento.min(0)
        maxs = treinamento.max(0)
        p = shape(treinamento)[1]
    fontes = zeros((np,sum(nProts)+1,shape(treinamento)[1]))
    pesosFontes = ones((np,sum(nProts),p))
    classesFontes = zeros((np,sum(nProts)))
    fitness = zeros(np)
    limites = zeros(np)
    for fonte in range(np):
        [fontes[fonte], classesFontes[fonte], pesosFontes[fonte]] = gerarSolucao(treinamento, classesTreinamento, mins, maxs, nProts, intervalar)
        if fonte == 0:
            [fontes[fonte], pesosFontes[fonte]] = refinar(treinamento, classesTreinamento, fontes[fonte], classesFontes[fonte], pesosFontes[fonte], intervalar)
        fitness[fonte] = calcularCriterioJ(pesosFontes[fonte], fontes[fonte], classesFontes[fonte], treinamento, classesTreinamento, p, intervalar)
        
    indice = argmax(fitness)
    GBEST = copy(fontes[indice])
    classesGBEST = copy(classesFontes[indice])
    fitnessGBEST = fitness[indice]
    pesosGBEST = copy(pesosFontes[indice])
    return [pesosFontes, fontes, classesFontes, GBEST, classesGBEST, fitnessGBEST, pesosGBEST, mins, maxs, fitness, limites]

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
    #gama = 0.3
    
    erro = testar(treinamento, classesTreinamento, particula, classesParticula, pesos, intervalar)
    
    n = shape(treinamento)[0]
    k = shape(particula)[0]-1
    distancias = calcularDistancias(particula[1:], treinamento, pesos, particula[0,:p], n, k, intervalar)
    graus = fcm.calcularGraus(distancias)
    criterio = fcm.calcularCriterio(graus, distancias, classesTreinamento, classesParticula)
    
    #selecionadas = selecionarConsideradas(treinamento, particula[0,:p], intervalar)
    #prodCorrInterna = prod(abs(matrizDeCorrelacao(selecionadas, intervalar)))
    
    return 1.0 / (1 + alfa * erro + beta * criterio )#+ gama * prodCorrInterna)
 
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

def ajustarVariavelFonte(fonte, prot, variavel, minimo, maximo, p, intervalar = True):
    if prot == 0 or not intervalar:
        if fonte[prot,variavel] < minimo:
            fonte[prot,variavel] = minimo
        elif fonte[prot,variavel] > maximo:
            fonte[prot,variavel] = maximo
    elif intervalar:
        if fonte[prot,variavel] < minimo:
            fonte[prot,variavel] = minimo
        elif fonte[prot,variavel] > maximo:
            fonte[prot,variavel] = maximo
        if variavel % 2 == 0:
            if fonte[prot,variavel] > fonte[prot,variavel + 1]:
                temp = fonte[prot,variavel]
                fonte[prot,variavel] = fonte[prot,variavel + 1]
                fonte[prot,variavel + 1] = temp
        else:
            if fonte[prot,variavel] < fonte[prot,variavel - 1]:
                temp = fonte[prot,variavel]
                fonte[prot,variavel] = fonte[prot,variavel - 1]
                fonte[prot,variavel - 1] = temp
    if prot == 0:        
        while sum(around(fonte[0,:p])) == 0.0:
            fonte[0,:p] = random.rand(p)
    return fonte   

def enviarAbelhas(tipo, nFontes, fontes, classesFontes, pesosFontes, mins, maxs, fitness, limites, treinamento, classesTreinamento, intervalar = True):
    pa = shape(treinamento)[1]/2
    if not intervalar:
        pa = shape(treinamento)[1]
    nDimensao = shape(treinamento)[1]
    k = shape(fontes[0])[0] - 1
    n = shape(treinamento)[0]
    probs = fitness / sum(fitness)
    for i in range(nFontes):  
        consideradasAntes = copy(fontes[i,0,:pa])
        ki = i 
        while ki==i:
            if tipo == "observadoras":
                ki = random.choice(int(nFontes),p=probs)
            else:
                ki = random.randint(nFontes)
        prot = random.randint(k)
        u = random.randint(nDimensao)
        while intervalar and prot == 0 and u in range(pa,2*pa):
            u = random.randint(nDimensao)
        phi = random.rand()*2 - 1
        anterior = fontes[i,prot,u]
        fontes[i,prot,u] = fontes[i,prot,u] + phi*(fontes[i,prot,u] - fontes[ki,prot,u])
        
        if intervalar:
            minimo = mins[u/2]
            maximo = maxs[u/2]
        else:
            minimo = mins[u]
            maximo = maxs[u]
        if prot == 0:
            minimo = 0
            maximo = 1
        fontes[i] = ajustarVariavelFonte(fontes[i], prot, u, minimo, maximo, pa, intervalar)
        
        pe = copy(pesosFontes[i])
        if any(around(consideradasAntes) != around(fontes[i,0,:pa])):
            pe = []
        pesos = calcularPesosPrototipos(treinamento, classesTreinamento, fontes[i,1:], classesFontes[i], fontes[i,0,:pa], n, k, pa, intervalar, pe)
            
        f = calcularCriterioJ(pesos, fontes[i], classesFontes[i], treinamento, classesTreinamento, pa, intervalar)
        if f > fitness[i]:
            #print "ajustou"
            pesosFontes[i] = pesos
            fitness[i] = f
            limites[i] = 0
        else:
            #print "nao ajustou"
            fontes[i,prot,u] = anterior
            limites[i] = limites[i] + 1
    return [fontes, pesosFontes, fitness, limites]

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

def enviarAbelhasEscoteiras(nFontes, fontes, classesFontes, pesosFontes, mins, maxs, fitness, limites, treinamento, classesTreinamento, limite, nProts, intervalar = True):
    ind = argmax(limites)
    if limites[ind] >= limite:     
        p = shape(treinamento)[1]/2
        if not intervalar:
            p = shape(treinamento)[1]   
        [va, pesos] = refinar(treinamento, classesTreinamento, fontes[ind], classesFontes[ind], pesosFontes[ind], intervalar)
        f = calcularCriterioJ(pesos, va, classesFontes[ind], treinamento, classesTreinamento, p, intervalar)
        if f > fitness[ind]:
            #print "refinou"
            fitness[ind] = f
            limites[ind] = 0
            fontes[ind] = va
            pesosFontes[ind] = pesos
        else:
            #print "gerou nova"
            [fontes[ind], classesFontes[ind], pesosFontes[ind]] = gerarSolucao(treinamento, classesTreinamento, mins, maxs, nProts, intervalar)
            
            fitness[ind] = calcularCriterioJ(pesosFontes[ind], fontes[ind], classesFontes[ind], treinamento, classesTreinamento, p, intervalar)
            limites[ind] = 0
    return [fontes, pesosFontes, fitness, limites]

def fazerCrossover(nFontes, fontes, classesFontes, pesosFontes, mins, maxs, fitness, limites, treinamento, classesTreinamento, intervalar = True):
    nPais = fix(nFontes / 2)
    pa = shape(treinamento)[1]/2
    if not intervalar:
        pa = shape(treinamento)[1]
    nDimensao = prod(shape(fontes[0]))
    k = shape(fontes[0])[0] - 1
    n = shape(treinamento)[0]
    pais  = zeros((nPais, nDimensao))
    fPais = zeros(nPais)
    probs = fitness / sum(fitness)
    for i in range (nPais):        
        k1 = random.choice(int(nFontes),p=probs)
        k2 = random.choice(int(nFontes),p=probs)
        if fitness[k1] > fitness[k2]:
            pais[i,:] = fontes[k1].ravel()
            fPais[i] = fitness[k1]
        else:
            pais[i,:] = fontes[k2].ravel()
            fPais[i] = fitness[k2]
    pPais = fPais / sum(fPais)
    for i in range(nFontes):
        genitor1 = random.choice(int(nPais),p=pPais)
        genitor2 = genitor1
        while genitor2 == genitor1:
            genitor2 = random.choice(int(nPais),p=pPais)
        prole = random.rand()* pais[genitor1,:] + random.rand()* pais[genitor2,:]
        
        prole = ajustarMinsMaxs(ajustarVariaveisConsideradas(reshape(prole,shape(fontes[i])),pa), mins, maxs, intervalar)
        
        pesos = calcularPesosPrototipos(treinamento, classesTreinamento, prole[1:], classesFontes[i], prole[0,:pa], n, k, pa, intervalar)
            
        f = calcularCriterioJ(pesos, prole, classesFontes[i], treinamento, classesTreinamento, pa, intervalar)
        
        if f > fitness[i]:
            #print "crossovou"
            fontes[i] = prole
            pesosFontes[i] = pesos
            fitness[i] = f
            limites[i] = 0
        else:
            limites[i] = limites[i] + 1
    return [fontes, pesosFontes, fitness, limites]

def memorizarMelhorSolucao(fitness, fontes, classesFontes, pesosFontes, GBEST, fitnessGBEST, classesGBEST, pesosGBEST):
    indice = argmax(fitness)
    if fitness[indice] > fitnessGBEST:
        GBEST = copy(fontes[indice])
        classesGBEST = copy(classesFontes[indice])
        fitnessGBEST = fitness[indice]
        pesosGBEST = copy(pesosFontes[indice])
    return [GBEST, classesGBEST, fitnessGBEST, pesosGBEST]

def treinar(mins, maxs, particulas, classesParticulas, pesosParticulas, GBEST, classesGBEST, fitnessGBEST, pesosGBEST, treinamento, classesTreinamento, nAbelhas, fitness, limites, nProts, intervalar = True):
    p = shape(treinamento)[1]/2
    if not intervalar:
        p = shape(treinamento)[1]
    k = shape(GBEST)[0]-1
    limite = 10
    np = fix(nAbelhas/2)
           
    MAX_ITERACAO_TOTAL = 1000
    epsilon = 0.00001
    Jatual = 1
    Jdepois = -1
    r = 0
    manteve = 0
    while r < MAX_ITERACAO_TOTAL and manteve < 5*limite:
        [particulas, pesosParticulas, fitness, limites] = enviarAbelhas("trabalhadoras", np, particulas, classesParticulas, pesosParticulas, mins, maxs, fitness, limites, treinamento, classesTreinamento, intervalar)
        [particulas, pesosParticulas, fitness, limites] = enviarAbelhas("observadoras", np, particulas, classesParticulas, pesosParticulas, mins, maxs, fitness, limites, treinamento, classesTreinamento, intervalar)
        [particulas, pesosParticulas, fitness, limites] = fazerCrossover(np, particulas, classesParticulas, pesosParticulas, mins, maxs, fitness, limites, treinamento, classesTreinamento, intervalar)
        [particulas, pesosParticulas, fitness, limites] = enviarAbelhasEscoteiras(np, particulas, classesParticulas, pesosParticulas, mins, maxs, fitness, limites, treinamento, classesTreinamento, limite, nProts, intervalar)
        
        [GBEST, classesGBEST, fitnessGBEST, pesosGBEST] = memorizarMelhorSolucao(fitness, particulas, classesParticulas, pesosParticulas, GBEST, fitnessGBEST, classesGBEST, pesosGBEST)
        Jatual = Jdepois     
        Jdepois = fitnessGBEST
        if abs(Jatual - Jdepois) <= epsilon:
            manteve = manteve + 1
        else:
            manteve = 0
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
            [pesosFontes, fontes, classesFontes, GBEST, classesGBEST, fitnessGBEST, pesosGBEST, mins, maxs, fitness, limites] = inicializar(treinamento, classesTreinamento, nProts, np, intervalar)
            [GBEST, classesGBEST, pesosGBEST] = treinar(mins, maxs, fontes, classesFontes, pesosFontes, GBEST, classesGBEST, fitnessGBEST, pesosGBEST, treinamento, classesTreinamento, np, fitness, limites, nProts, intervalar)
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
    print "fcminicio0.50.5"
    print "mediterraneo_oceanico_normalizados_limpo.txt"
    [dados, classes] = lerDados("mediterraneo_oceanico_normalizados_limpo.txt", True)
    rodarValidacaoCruzada(dados, classes, [20, 40], 10, 10, 20, True)      