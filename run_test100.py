from experiments import *
from algorithms import *
from loggers import *
import pandas
import sys
import itertools as it


#parametri dell'esperimento
instanceFolder = 'instances_test'
outputFolder = 'results'
repetitions = 10
budget = 100
njobs = -1 #tutti i core disponibili
traj = False

#lettura del nome esperimento (sarà il nome file dei risultati)
if len(sys.argv)<2:
    print('USAGE: python run_experiment.py EXP_NAME [traj]')
    print('       EXP_NAME: it is the name of the experiment and it is used for generating the output filename.')
    print('       traj: if you pass the ''traj'' argument, the trajectories output file is generated.')
    sys.exit(0)
expName = sys.argv[1]
if len(sys.argv)>2:
    traj = sys.argv[2]!='no' #salva traiettorie o no

#creazione esperimento
exp = Experiment(instanceFolder,repetitions,budget,expName)

#definizione algoritmi da sperimentare
alg1 = FATRLS(tabuSize='full',iniDistance='half',beta=1.2) #valori default
exp.addAlgorithm(alg1)

#imposta il logger
logger = ExperimentLogger()
exp.addLogger(logger)
if traj: #salva traiettorie
    exp.addLogger(FullExecutionLogger())

#esegui l'esperimento
exp.run(njobs)

#salva i risultati mettendoli insieme a quelli di UMM e CEGO
df = logger.getDataFrame()
df.to_pickle(f'{outputFolder}/{expName}.pickle')
df.to_csv(f'{outputFolder}/{expName}.csv', sep=';', index=False)

#salva i dati delle traiettorie se richiesto
if traj:
    df = logger.getExecutionDataFrames()['FullExecutionLogger']
    df.to_pickle(f'{outputFolder}/{expName}_traj.pickle')
    df.to_csv(f'{outputFolder}/{expName}_traj.csv', sep=';', index=False)
