from numpy import *



def mediaIntervalar(a, b):
    n = len(a)
    return sum(a + b) / (2. * n)

def varianciaIntervalar(a, b):
    n = len(a)
    fator1 = sum(a * a + a * b + b * b) / (3. * n)
    fator2 = sum(a + b) ** 2. / (4 * n ** 2.)
    return fator1 - fator2
  
def covarianciaIntervalar(a, b, c, d):
    n = len(a)
    x1 = mediaIntervalar(a, b)
    x2 = mediaIntervalar(c, d)
    return sum(2 * (a - x1) * (c - x2) + (a - x1) * (d - x2) + (b - x1) * (c - x2) + 2 * (b - x1) * (d - x2)) / (6 * n)   

def matrizDeCorrelacao(variaveis, intervalar = True):
    if intervalar:
        mins = variaveis[:,::2]
        maxs = variaveis[:,1::2]
        p = shape(mins)[1]
        matriz = zeros((p,p))
        for var1 in range(p):
            for var2 in range(p):
                a = mins[:,var1]
                b = maxs[:,var1]
                c = mins[:,var2]
                d = maxs[:,var2]
                matriz[var1,var2] = covarianciaIntervalar(a, b, c, d) / (sqrt(varianciaIntervalar(a, b))*sqrt(varianciaIntervalar(c, d)))
        return matriz
    else:
        return corrcoef(variaveis, rowvar=0)

def foo():
    global a 
    a = False    

def foo2():
    if a:
        print "a"
  
if __name__ == "__main__":
    foo()
    foo2()