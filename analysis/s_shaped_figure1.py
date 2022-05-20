import numpy as np
import matplotlib.pyplot as plt

def s(x,beta,d_ini):
    #return d_ini * ( 1 - 1 / ( 1 + (x/(1-x))**(-beta) ) )
    return 1 + (d_ini-1) * ( 1 - 1 / ( 1 + (x/(1-x))**(-beta) ) )

x = np.linspace(1e-6,1-1e-6,1000)
d_ini = 25
s10 = s(x,1.0,d_ini)
#s15 = s(x,1.5,d_ini)
s20 = s(x,2.0,d_ini)
s30 = s(x,3.0,d_ini)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xmargin(0)
ax.set_ymargin(0)
ax.set_xlim(xmin=0.,xmax=1.)
ax.set_ylim(ymin=0.,ymax=d_ini)
ax.set_xlabel('Evolution percentage $p$')
ax.set_ylabel('Perturbation strength $d$')

ax.plot(x,s10,label=r'$\beta=1$')
ax.plot(x,s20,label=r'$\beta=2$',dashes=(4,1))
ax.plot(x,s30,label=r'$\beta=3$',dashes=(1,1))

ax.legend(loc='upper right')

yticks = list(range(0,d_ini+1,5))
yticks[0] = 1
yticks_labels = [ str(y) for y in range(0,d_ini+1,5) ]
yticks_labels[0] = '1'
yticks_labels[-1] = f'$d_\\mathrm{{ini}} = {yticks_labels[-1]}$'
plt.yticks(ticks=yticks, labels=yticks_labels)

plt.show()