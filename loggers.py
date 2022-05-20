import numpy as np
import pandas as pd





class AbstractLogger:

    def __call__(self, **kwargs):
        '''
        To be defined in concrete subclasses.
        '''
        self.loggerType = 'no' #should be execution or experiment
        pass

    def reset(self):
        '''
        To be called for multiple use of the same logger.
        To be defined in concrete subclasses.
        '''
        pass

    def getDataFrame(self):
        '''
        Returns the pandas dataframe with the results.
        After this call, stop to use the logger.
        To be defined in concrete subclasses.
        '''
        pass





class ExecutionLogger(AbstractLogger):

    def __init__(self):
        self.loggerType = 'execution'
        self.reset()
        #done

    def __call__(self, **kwargs):
        best_fx = kwargs['best_fx']
        best_nfev = kwargs['best_nfev']
        if best_fx<self.previous_best_fx: #altrimenti è uguale, non può essere maggiore!
            self.rows.append({  'nfev':     best_nfev,
                                'fitness':  best_fx     })
        self.previous_best_fx = best_fx
        #done

    def reset(self):
        self.previous_best_fx = np.inf
        self.rows = []
        #done

    def getDataFrame(self):
        return pd.DataFrame(self.rows)
        #done





class FullExecutionLogger(AbstractLogger):

    def __init__(self):
        self.loggerType = 'execution'
        self.reset()
        #done

    def __call__(self, **kwargs):
        cur_fx,cur_nfev = kwargs['cur_fx'],kwargs['cur_nfev']
        self.rows.append({'nfev': cur_nfev, 'fitness': cur_fx})
        #done

    def reset(self):
        self.rows = []
        #done

    def getDataFrame(self):
        return pd.DataFrame(self.rows)
        #done





class ExperimentLogger(AbstractLogger):

    def __init__(self):
        self.loggerType = 'experiment'
        self.reset()
        #done

    def __call__(self, **kwargs):
        #tirare fuori i dataframe dalle chiavi _df dal dizionario kwargs, aggiungergli attributi problem,instance,n,algorithm,budget,parameters,run e poi metterli nella lista executions
        keys = list(kwargs.keys())
        for key in keys:
            if key[-3:]=='_df':
                df = kwargs.pop(key)
                df['problem'] = kwargs['problem']
                df['instance'] = kwargs['instance']
                df['n'] = kwargs['n']
                df['algorithm'] = kwargs['algorithm']
                df['budget'] = kwargs['budget']
                df['parameters'] = kwargs['parameters']
                df['run'] = kwargs['run']
                ekey = key[:-3]
                if ekey not in self.executions: self.executions[ekey] = []
                self.executions[ekey].append(df)
        #aggiungere alle rows il dizionario kwargs senza le chiavi _df
        self.rows.append(kwargs)
        #done

    def reset(self):
        self.rows = []
        self.executions = {} #dizionario con chiave = nome execution logger e valore = lista di dataframes
        #done

    def getDataFrame(self):
        return pd.DataFrame(self.rows)
        #done

    def getExecutionDataFrames(self):
        #ritornare un dizionario come executions ma con i valori (liste di dataframes) contratti in un unico dataframe
        for key in self.executions:
            self.executions[key] = pd.concat(self.executions[key])
        return self.executions
        #done
