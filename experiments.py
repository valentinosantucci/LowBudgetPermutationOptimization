from problems import *
import os
import sys
from joblib import Parallel, delayed
import itertools as it
import time





def createObjectiveFunction(problem, filename):
    if problem=='lop':
        return LOP_ObjectiveFunction(filename)
    elif problem=='pfsp':
        return PFSP_ObjectiveFunction(filename)
    elif problem=='qap':
        return QAP_ObjectiveFunction(filename)
    else:
        print(f'ERROR: {problem} is not a valid name for a problem!',file=sys.stderr)
        sys.exit(1)
    #done





class Experiment:

    def __init__(self, instanceFolder, repetitions, budget, name='unnamed'):
        '''
        Initialize an experiment.
        - instanceFolder: folder made by one subfolder for any problem, each subfolder should contain instance files
        - repetitions: is the number of runs per algorithm per instance
        - budget: number of evaluations for each execution
        - name: name given to the experiment
        '''
        self.instanceFolder = instanceFolder
        self.repetitions = repetitions
        self.budget = budget
        self.name = name
        self.algorithms = []
        self.loggers = []
        #done

    def addAlgorithm(self, algorithm):
        self.algorithms.append(algorithm)
        #done

    def addLogger(self, logger):
        self.loggers.append(logger)
        #done

    def _process(self,algorithm,objfun,run):
        #print to screen
        print(f'* Run #{run} of {algorithm.getName()}({algorithm.getParameters()}) on {objfun.instance} started ...', file=sys.stderr)
        #fai un run
        objfun.reset()
        startTime = time.time()
        algorithm.minimize(objfun, budget=self.budget, seed=run)
        endTime = time.time()
        executionTime = int((endTime-startTime)*1000+0.5) #in millisecondi
        #print to screen
        print(f'# Run #{run} of {algorithm.getName()}({algorithm.getParameters()}) on {objfun.instance} finished.', file=sys.stderr)
        #costruisci dizionario da ritornare
        result = {'problem':      objfun.problem,
                  'instance':     objfun.instance,
                  'n':            objfun.n,
                  'algorithm':    algorithm.getName(),
                  'budget':       self.budget,
                  'parameters':   algorithm.getParameters(),
                  'run':          run,
                  'fitness':      objfun.best_fx,
                  'nfev':         objfun.best_nfev,
                  'time':         executionTime }
        #recupera dataframe degli execution loggers registrati e aggiungili al dizionario dei risultati
        for logger in objfun.loggers:
            loggerName = str(type(logger))
            loggerName = loggerName[loggerName.rindex('.')+1:loggerName.rindex("'")]
            df = logger.getDataFrame()
            result[f'{loggerName}_df'] = df
        #ritorna il dizionario
        return result
        #done

    def run(self, njobs=-1):
        #crea lista di objective functions (con gli execution loggers agganciati)
        objfuns = []
        for dir in os.scandir(self.instanceFolder):
            if not dir.is_dir(): continue
            problem = dir.name
            for file in os.scandir(dir.path):
                if not file.is_file(): continue
                filename = file.path
                objfun = createObjectiveFunction(problem,filename)
                for logger in self.loggers:
                    if logger.loggerType!='execution': continue
                    objfun.addLogger(logger)
                objfuns.append(objfun)
        #lancia i processi in parallelo: un processo per ogni run
        for result in Parallel(n_jobs=njobs)( delayed(self._process)(algorithm,objfun,run)
                                              for algorithm,objfun,run
                                              in it.product(self.algorithms,objfuns,range(1,self.repetitions+1)) ):
            for logger in self.loggers:
                if logger.loggerType!='experiment': continue
                logger(**result)
        #done