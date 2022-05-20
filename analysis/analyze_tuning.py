import pandas as pd
import sys
from scipy.stats import friedmanchisquare
import scikit_posthocs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np


#leggi file dei risultati
filename = '../results/tuning_results.pickle'
'''if len(sys.argv)<2:
    print('USAGE: python analyze_test.py [TUNING_RESULTS_PICKLE_FILE]')
    sys.exit()'''
if len(sys.argv)>=2: filename = sys.argv[1]
df = pd.read_pickle(filename)

#crea nuove colonne con valori dei singoli parametri
def get_k(row):
    s = row.parameters.split(',')[0].split('=')[1]
    return '$n$' if s=='full' else '$0.75n$' if s=='threequarters' else '$0.5n$' if s=='half' else '$0.25n$' if s=='quarter' else '0'
df['k'] = df.apply(get_k,axis=1)
def get_dini(row):
    s = row.parameters.split(',')[1].split('=')[1]
    return '$0.5n$' if s=='half' else '$0.25n$'
df['dini'] = df.apply(get_dini,axis=1)
def get_beta(row):
    return row.parameters.split(',')[2].split('=')[1]
df['beta'] = df.apply(get_beta,axis=1)

#crea colonna degli rpd
algs = df.parameters.unique()
instances = df.instance.unique()
runs = df.run.unique()
idf = df.set_index(['parameters','instance','run'])
idf['rpd'] = -1.
for instance in instances:
    best = df[df.instance==instance].fitness.min()
    for run in runs:
        for alg in algs:
            fit = idf.loc[(alg,instance,run)].fitness
            rpd = 100*(fit-best)/best
            idf.at[(alg,instance,run),'rpd'] = rpd
df = idf.reset_index()

#crea tabella pivot degli arpd
pt_arpd = df.pivot_table(index='parameters', columns='instance', values='rpd', aggfunc='mean', margins=True, margins_name='Overall ARPD')
pt_arpd = pt_arpd[:-1] #rimuove il totale delle righe
pt_arpd = pt_arpd.sort_values('Overall ARPD')

#crea tabella pivot dei rank degli arpd
pt_rank = pt_arpd.rank()
del pt_rank['Overall ARPD']
pt_rank['Avg Rank'] = pt_rank.mean(axis=1)
pt_rank = pt_rank.sort_values('Avg Rank')

#unisci overall arpd e avg rank in un unico dataframe ordinato per avg rank
t = pd.concat( [pt_rank['Avg Rank'],pt_arpd['Overall ARPD']], axis=1 )
t = t.sort_values('Avg Rank')

#esegui test di friedman sugli arpd e stampa risultato
print( friedmanchisquare(*pt_arpd.to_numpy()) )

#esegui analisi posthoc di setting con best_avgrank contro tutti gli altri
#usa metodo posthoc di conover
#inserisci pvalue nel dataframe t
winner = t.iloc[0].name
pvalues = scikit_posthocs.posthoc_conover_friedman(pt_arpd.T)
pvalues = pvalues.loc[winner]
t['Post-hoc p-value'] = t.apply( lambda row: pvalues[row.name], axis=1 )

#stampa tutti i p-values della posthoc
pos = 0
for row in t.iterrows():
    pos += 1
    print(f'#{pos} \t {row[0]} \t {row[-1]["Post-hoc p-value"]}')

#crea latex in output
'''
def parameters2setting(params):
    #inizio
    s = params.split(',')
    r = '' #r = r'$\left('
    #d_ini
    r += r'd_\mathrm{ini}='
    v = s[1].split('=')[1]
    r += r'0.5n' if v=='half' else r'0.25n'
    #beta
    r += r';\,\,\beta='
    v = s[2].split('=')[1]
    r += v
    #k
    r += r';\,\,k='
    v = s[0].split('=')[1]
    r += r'n' if v=='full' else r'0.75n' if v=='threequarters' else r'0.5n' if v=='half' else r'0.25n' if v=='quarter' else '0'
    #fine
    #r += r'\right)$'
    return r
t.rename(index=parameters2setting, inplace=True)
t.index.rename('Setting', inplace=True)
'''
array_dini = [ s.split(',')[1].split('=')[1] for s in t.index]
array_dini = [ '$0.5n$' if s=='half' else '$0.25n$' for s in array_dini ]
array_beta = [ s.split(',')[2].split('=')[1] for s in t.index]
array_k    = [ s.split(',')[0].split('=')[1] for s in t.index]
array_k    = [ '$n$' if s=='full' else '$0.75n$' if s=='threequarters' else '$0.5n$' if s=='half' else '$0.25n$' if s=='quarter' else '0' for s in array_k ]
t.index = pd.MultiIndex.from_arrays([array_dini,array_beta,array_k])
t.rename_axis((r'$d_\mathrm{ini}$',r'$\beta$',r'$k$'), inplace=True)
with pd.option_context("max_colwidth", 1000):
    t.to_latex( 'tab_tuning.tex', 
                columns=['Avg Rank', 'Overall ARPD'],
                header=['Avg Rank', 'Overall ARPD'],
                float_format='%.2f',
                escape=False,
                sparsify=False,
                )

#crea i 3 boxplot per ogni parametro
'''
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
np.set_printoptions(linewidth=np.nan)
sns.set_style("whitegrid")
'''
plt.figure()
ax = sns.boxplot(data=df, x='dini', y='rpd', whis=1.5, width=0.6, fliersize=0, order=['$0.25n$','$0.5n$'])
ax.set_xlabel(r'$d_\mathrm{ini}$')
ax.set_ylabel(r'$\mathit{rpd}$') #Relative Percentage Deviation
ax.set_ylim(-1,36)
#ax.set_title(r'Initial Distance $d_\mathrm{ini}$', fontweight='bold')
plt.tight_layout()
plt.savefig('boxplot_dini.pdf')
plt.figure()
ax = sns.boxplot(data=df, x='beta', y='rpd', whis=1.5, width=0.6, fliersize=0)
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\mathit{rpd}$')
ax.set_ylim(-1,36)
#ax.set_title(r'Exponent $\beta$', fontweight='bold')
plt.tight_layout()
plt.savefig('boxplot_beta.pdf')
plt.figure()
ax = sns.boxplot(data=df, x='k', y='rpd', whis=1.5, width=0.6, fliersize=0, order=['0','$0.25n$','$0.5n$','$0.75n$','$n$'])
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\mathit{rpd}$')
ax.set_ylim(-1,36)
#ax.set_title(r'Tabu Size $k$', fontweight='bold')
plt.tight_layout()
plt.savefig('boxplot_k.pdf')
