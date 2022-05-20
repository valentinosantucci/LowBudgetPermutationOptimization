import numpy as np
import os
import sys


class AbstractPermutationObjectiveFunction:

    def __init__(self, problem, instance):
        '''
        Initialize an abstract objective function.
        - problem: problem name (eg: lop, pfsp, ...)
        - instance: instance name
        '''
        self.problem = problem
        self.instance = instance
        self.n = None
        self.loggers = []
        self.reset()
        #done

    def __call__(self, x):
        '''
        Evaluate the objective function.
        Never redefine this in subclasses!
        - x is a permutation (numpy int64 array)
        - the returned value is the objective function evaluated in x
        '''
        fx = self._fitness(x)
        self.nfev += 1
        if fx<self.best_fx:
            self.best_x = x.copy()
            self.best_fx = fx
            self.best_nfev = self.nfev
        for logger in self.loggers:
            logger( best_x =    self.best_x,
                    best_fx =   self.best_fx,
                    best_nfev = self.best_nfev,
                    cur_x =     x,
                    cur_fx =    fx,
                    cur_nfev =  self.nfev )
        return fx
        #done

    def getSize(self):
        '''
        Size of the instance. It may be a single integer or a tuple of integers depending from the problem.
        '''
        return -1
        #done

    def addLogger(self, logger):
        '''
        Add a logger instance to the objective function.
        - logger: should be of type AbstractLogger or subclass
        '''
        if logger.loggerType!='execution':
            print('ERROR: The logger should be of type execution!', file=sys.stderr)
            sys.exit(1)
        self.loggers.append(logger)
        #done

    def reset(self):
        '''
        Reset nfev and all the registered loggers for multiple runs of the same objective function.
        '''
        self.nfev = 0
        self.best_x = None
        self.best_fx = np.inf
        self.best_nfev = None
        for logger in self.loggers:
            logger.reset()
        #done

    def _fitness(self,x):
        '''
        True fitness computation. To be redefined by subclasses.
        '''
        pass
        #done





class LOP_ObjectiveFunction(AbstractPermutationObjectiveFunction):

    def __init__(self, filename):
        '''
        Initialize a LOP objective function for minimization.
        - filename: name of the instance file to read from
        '''
        super().__init__( problem='lop', instance=os.path.basename(filename) )
        self.filename = filename
        self.B = None #matrix for lop instance
        self._readInstance()
        #done

    def _fitness(self, x):
        '''
        LOP objective function for minimization (without considering the diagonal of the instance matrix).
        - x should be a permutation
        - the returned value is the objective function evaluated in x
        '''
        fx = 0
        for i in range(1,self.n):
            for j in range(i):
                fx += self.B[x[i]][x[j]]
        return fx
        #done

    def getSize(self):
        '''
        The size of a LOP instance is n, the size of the square matrix.
        '''
        return self.n
        #done

    def _readInstance(self):
        fp = open(self.filename)
        line = fp.readline()
        values = line.split()
        self.n = int(values[0])
        self.B = np.empty( (self.n,self.n), dtype='int32' )
        for i in range(self.n):
            line = fp.readline()
            values = line.split()
            for j in range(self.n):
                self.B[i][j] = int(values[j])
        fp.close()
        #done

    def convertoToFitnessWithDiagonal(self, fx):
        '''
        Convert a fitness computed for minization without diagonal to a fitness for minimization with diagonal.
        '''
        return fx + np.diag(self.B).sum()
        #done

    def convertToFitnessForMaximization(self, fx):
        '''
        Convert a fitness computed for minimization without diagonal to a fitness for maximization without diagonal.
        '''
        return self.B.sum() - np.diag(self.B).sum() - fx
        #done





class PFSP_ObjectiveFunction(AbstractPermutationObjectiveFunction):

    def __init__(self, filename):
        '''
        Initialize a PFSP objective function for minimization of the makespan.
        - filename: name of the instance file to read from
        '''
        instance = os.path.basename(filename)
        instance = instance[:instance.rindex('.')]
        super().__init__( problem='pfsp', instance=instance )
        self.filename = filename
        self.m = -1 #n already declared in superclass
        self.P = None #matrix for pfsp instance
        self._readInstance()
        #done

    def _fitness(self, x):
        '''
        PFSP objective function for the makespan criterion
        - x should be a permutation
        - the returned value is the objective function evaluated in x
        '''
        n,m,P = self.n,self.m,self.P
        c = np.zeros(m,dtype='int32')
        for i in range(n):
            for j in range(m):
                maxTerm = max(c[j],c[j-1]) if j>0 else c[j]
                c[j] = P[x[i],j] + maxTerm
        return c[-1]
        #done

    def getSize(self):
        '''
        The size of PFSP instance is (n,m), i.e. the size of the instance matrix.
        '''
        return (self.n,self.m)
        #done

    def _readInstance(self):
        if self.instance[:3]=='rec':
            self._readInstance_rec()
        else:
            print(f'Format not supported for the PFSP file {self.filename}', file=sys.stderr)
            sys.exit(1)
        #done

    def _readInstance_rec(self):
        fp = open(self.filename)
        line = fp.readline() #discard 1st line
        line = fp.readline()
        values = line.split()
        self.n = int(values[0])
        self.m = int(values[1])
        self.P = np.empty( (self.n,self.m), dtype='int32' )
        for i in range(self.n):
            line = fp.readline()
            values = line.split()[1::2] #discard even columns
            for j in range(self.m):
                self.P[i][j] = int(values[j])
        fp.close()
        #done





class QAP_ObjectiveFunction(AbstractPermutationObjectiveFunction):

    def __init__(self, filename):
        '''
        Initialize a QAP objective function
        - filename: name of the instance file to read from
        '''
        instance = os.path.basename(filename)
        instance = instance[:instance.rindex('.')]
        super().__init__( problem='qap', instance=instance )
        self.filename = filename
        self.A = None #matrix n x n
        self.B = None #matrix n x n
        self._readInstance()
        #done

    def _fitness(self, x):
        '''
        QAP objective function
        - x should be a permutation
        - the returned value is the objective function evaluated in x
        '''
        return np.sum(self.A * self.B[np.ix_(x,x)])
        #done

    def getSize(self):
        '''
        The size of QAP instance is n
        '''
        return self.n
        #done

    def _readInstance(self):
        f = open(self.filename,'r')
        self.n = int(f.readline().strip())
        self.A = np.loadtxt(f, max_rows=self.n, dtype='int32')
        self.B = np.loadtxt(f, max_rows=self.n, dtype='int32')
        f.close()
        #done

