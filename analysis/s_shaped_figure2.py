import numpy as np
import matplotlib.pyplot as plt

# Use Type 1 fonts in plots.
import matplotlib
import seaborn as sns
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
# Do not wrap long lines
np.set_printoptions(linewidth=np.nan)
sns.set_style("whitegrid")

def s(x,beta):
    return 1 - 1 / ( 1 + (x/(1-x))**(-beta) )

x = np.linspace(1e-6,1-1e-6,1000)
d_ini = 25
s10 = s(x,1.0)
#s15 = s(x,1.5)
s20 = s(x,2.0)
s30 = s(x,3.0)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax.set_xmargin(0)
ax.set_ymargin(0)
ax.set_xlim(xmin=0.,xmax=1.)
ax.set_ylim(ymin=0.,ymax=1.)
ax.set_xlabel(r'$p$')
ax.set_ylabel(r'$s_\beta(p)$')

ax.plot(x,s10,label=r'$\beta=1$')
ax.plot(x,s20,label=r'$\beta=2$',dashes=(4,1))
ax.plot(x,s30,label=r'$\beta=3$',dashes=(1,1))

ax.legend(loc='upper right')

plt.show()