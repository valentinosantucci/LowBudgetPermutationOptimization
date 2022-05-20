import pandas as pd
import sys
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon
import scikit_posthocs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np


#leggi file dei risultati
filename = '../results/test_results.pickle'
'''if len(sys.argv)<2:
    print('USAGE: python analyze_test.py [TEST_RESULTS_PICKLE_FILE]')
    sys.exit()'''
if len(sys.argv)>=2: filename = sys.argv[1]
df = pd.read_pickle(filename)

#crea colonna degli rpd
algs = df.algorithm.unique()
budgets = df.budget.unique()
instances = df.instance.unique()
runs = df.run.unique()
idf = df.set_index(['algorithm','budget','instance','run'])
idf['rpd'] = -1.
for instance in instances:
    best = df[df.instance==instance].fitness.min()
    for run in runs:
        for alg in algs:
            for budget in budgets:
                fit = idf.loc[(alg,budget,instance,run)].fitness
                rpd = 100*(fit-best)/best
                idf.at[(alg,budget,instance,run),'rpd'] = rpd
df = idf.reset_index()

#crea tabella pivot degli arpd solo per budget 400
df400 = df[df.budget==400]
pt_arpd = df400.pivot_table(index=['problem','instance'], columns='algorithm', values='rpd', aggfunc='mean', margins=False)
pt_arpd = pt_arpd.reindex(columns=['fat_rls', 'cego-er0', 'umm-er0'])
pt_arpd = pt_arpd.rename(columns={'fat_rls': 'FAT-RLS', 'cego-er0': 'CEGO', 'umm-er0': 'UMM'})

#test wilcoxon da stampare a video
print('WILCOXON FAT-RLS vs CEGO')
print(wilcoxon(pt_arpd['FAT-RLS'],pt_arpd['CEGO']))
print('WILCOXON FAT-RLS vs UMM')
print(wilcoxon(pt_arpd['FAT-RLS'],pt_arpd['UMM']))
print('WILCOXON CEGO vs UMM')
print(wilcoxon(pt_arpd['CEGO'],pt_arpd['UMM']))

#overall arpd a video
print(f'OVERALL ARPD FAT-RLS: {pt_arpd["FAT-RLS"].mean()}')
print(f'OVERALL ARPD CEGO: {pt_arpd["CEGO"].mean()}')
print(f'OVERALL ARPD UMM: {pt_arpd["UMM"].mean()}')

#test statistici di mann whitney nella tabella arpd
win  = r'$\blacktriangle$'
loss = r'$\triangledown$'
draw = r'\phantom{$\blacktriangle$}' #draw = r'$=$'
for instance in instances:
    problem = df400[ (df400.algorithm=='fat_rls') & (df400.instance==instance) ].problem.unique()[0]
    fat = df400[ (df400.algorithm=='fat_rls') & (df400.instance==instance) ].fitness.to_numpy()
    cego = df400[ (df400.algorithm=='cego-er0') & (df400.instance==instance) ].fitness.to_numpy()
    umm = df400[ (df400.algorithm=='umm-er0') & (df400.instance==instance) ].fitness.to_numpy()
    fat_vs_cego = mannwhitneyu(fat,cego).pvalue
    fat_vs_umm = mannwhitneyu(fat,umm).pvalue
    if fat_vs_cego<=0.05:   res_vs_cego = win if fat.mean()<cego.mean() else loss
    else:                   res_vs_cego = draw
    if fat_vs_umm<=0.05:    res_vs_umm = win if fat.mean()<umm.mean() else loss
    else:                   res_vs_umm = draw
    arpd_fat = pt_arpd.loc[(problem,instance),'FAT-RLS']
    arpd_cego = pt_arpd.loc[(problem,instance),'CEGO']
    arpd_umm = pt_arpd.loc[(problem,instance),'UMM']
    pt_arpd.loc[(problem,instance),'FAT-RLS'] = f'{arpd_fat:.2f}'
    pt_arpd.loc[(problem,instance),'CEGO'] = f'{arpd_cego:.2f} {res_vs_cego}'
    pt_arpd.loc[(problem,instance),'UMM'] = f'{arpd_umm:.2f} {res_vs_umm}'

#esporta tabella in latex
pt_arpd.to_latex('tab_test.tex', escape=False, sparsify=True)
#METTERE GRASSETTI, ALLINEAMENTI A DESTRA E FARE AGGIUSTAMENTI NOMI MANUALMENTE SUL LATEX!!!

#funzione utile ad aggiustare spacing nei boxplot raggruppati. Presa da https://stackoverflow.com/questions/56838187/how-to-create-spacing-between-same-subgroup-in-seaborn-boxplot
from matplotlib.patches import PathPatch
def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)
                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

#creare il boxplot
df.replace({'fat_rls':'FAT-RLS', 'cego-er0':'CEGO', 'umm-er0':'UMM'}, inplace=True)
fig = plt.figure()
ax = sns.boxplot(data=df, x='budget', y='rpd', hue='algorithm', whis=1.5, width=0.6, fliersize=0, hue_order=['FAT-RLS','CEGO','UMM'])
adjust_box_widths(fig, 0.9)
ax.set_xlabel('Budget of evaluations')
ax.set_ylabel(r'$\mathit{rpd}$')
ax.set_ylim(-1,53)
ax.legend(title='Algorithms', loc='upper right')
plt.tight_layout()
plt.savefig('boxplot_budget.pdf')

#stampa a video mann whitney di tutti gli rpd su ogni budget
budgets = [100,200,400]
for budget in budgets:
    fat_rpd = df[ (df.budget==budget) & (df.algorithm=='FAT-RLS') ].rpd
    cego_rpd = df[ (df.budget==budget) & (df.algorithm=='CEGO') ].rpd
    umm_rpd = df[ (df.budget==budget) & (df.algorithm=='UMM') ].rpd
    print(f'--- BUDGET={budget} ---')
    print('MANNWHITNEY ALL-RPDs OF FAT-RLS vs CEGO: ', mannwhitneyu(fat_rpd,cego_rpd))
    print('MANNWHITNEY ALL-RPDs OF FAT-RLS vs UMM: ', mannwhitneyu(fat_rpd,umm_rpd))
    print('MANNWHITNEY ALL-RPDs OF CEGO vs UMM: ', mannwhitneyu(cego_rpd,umm_rpd))