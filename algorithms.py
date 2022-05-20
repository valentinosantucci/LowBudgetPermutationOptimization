import numpy as np
import random





def srand(seed):
    np.random.seed(seed)
    random.seed(seed)
    #done





class AbstractAlgorithm:

    def __init__(self):
        '''
        The concrete subclasses need to put parameters of the algorithm in the constructor.
        '''
        #done

    def minimize(self, objfun, budget, seed=0):
        '''
        Minimize the objective function using the given budget of evaluations.
        No need to return or log nothing, the loggers will do it.
        To be defined in a concrete subclass.
        '''
        pass
        #done

    def getName(self):
        '''
        It should return the name of the algorithm.
        To be defined in subclasses.
        '''
        pass
        #done

    def getParameters():
        '''
        It should return a string representation of the parameters of the algorithm.
        To be defined in subclasses.
        '''
        pass
        #done




######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
class RandomSearch(AbstractAlgorithm):
    def getName(self):          return 'RS'
    def getParameters(self):    return ''
    def minimize(self, objfun, budget, seed=0):
        srand(seed)
        for _ in range(budget):
            objfun( np.random.permutation(objfun.n) )
        #done





######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
class FATRLS(AbstractAlgorithm):
    
    def __init__(self, tabuSize='full', iniDistance='half', beta=1.2): #tunati, i default sono i meglio
        self.tabuSize = tabuSize        # 'full' or 'ninety' or 'threequarters' or 'half' or 'quarter' or 'zero'
        self.iniDistance = iniDistance  # 'half' or 'quarter'
        self.beta = beta
    
    def getName(self):          return  'fat_rls'
    def getParameters(self):    return f'ts={self.tabuSize},id={self.iniDistance},beta={self.beta}'
    
    def minimize(self, objfun, budget, seed=0):
        #funzioni
        s_shaped_function = lambda x: 1 / ( 1 + (x/(1-x))**(-self.beta) )
        #inizializza
        srand(seed)
        n = objfun.n
        ts = n if self.tabuSize=='full' else int(n*0.90) if self.tabuSize=='ninety' else int(n*0.75) if self.tabuSize=='threequarters' else n//2 if self.tabuSize=='half' else n//4 if self.tabuSize=='quarter' else 0
        id = n//2 if self.iniDistance=='half' else n//4
        tabu = []
        x = np.random.permutation(n)
        fx = objfun(x)
        #main loop
        for nfev in range(2,budget+1):
            #clona x nella offspring y
            y = x.copy()
            #decidi la mutation strength (distanza dell'inserzione)
            norm_nfev = np.clip(nfev/budget, 0.01, 0.99)
            d = int(np.round(id*(1-s_shaped_function(norm_nfev))))
            if d<1: d = 1
            #decidi l'inserzione i,j ... y[j] si sposta in posizione i
            i,j = -1,-1
            while j<0 or y[j] in tabu:
                i = np.random.randint(0,n-d)
                j = i+d
                if np.random.rand()<0.5: i,j = j,i
            #effettua l'inserzione i,j sulla permutazione y
            if i<j:
                temp = y[i:j].copy()
                y[i] = y[j]
                y[i+1:j+1] = temp
            else:
                temp = y[j+1:i+1].copy()
                y[i] = y[j]
                y[j:i] = temp
            #valuta la offspring y
            fy = objfun(y)
            #rimpiazza current individual x se il caso
            if fy<fx: x,fx = y,fy
            #aggiorna la tabu list
            tabu.insert(0,y[i])
            if len(tabu)>=ts: tabu.pop()
        #end main loop
        #done

