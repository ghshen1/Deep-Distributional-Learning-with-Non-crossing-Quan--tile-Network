from generate import gen_univ
from generate import quant_univ
import torch
import matplotlib.pyplot as plt

# Generate data from the ground truth models

SIZE=2**9; taus=torch.Tensor([0.05,0.25,0.5,0.75,0.95]).unsqueeze(1);
models=['linear','wave','angle']
errors=['t','expx_t','sinex_t']
modelnames=['Linear','Wave','Angle']
errornames=[r'$t(1)$',r'Exp-$t(2)$',r'Sine-$t(2)$']
dataset=[];
x0=torch.linspace(0,1,1000).unsqueeze(dim=1);x=[];y=[];quant0=[]
for j in range(len(errors)):
        dataset=gen_univ(size=SIZE,model=models[j],error=errors[j],df=2,sigma=1)
        x.append(dataset[:][0].data.numpy())
        y.append(dataset[:][1].data.numpy())
        quant0.append(quant_univ(x0,taus,model=models[j],error=errors[j],df=2,sigma=1))
ylims=[[-4,7],[-7,7],[0,7.5]]
positions=['upper left','lower left','upper left']
#colors=['r','darkorange','lime','darkviolet','deepskyblue']
names=(r'$\tau=0.05$',r'$\tau=0.25$',r'$\tau=0.5$',r'$\tau=0.75$',r'$\tau=0.95$','Data')

# Plot and show the ground truth models and their conditional quantile curves
figs, axs = plt.subplots(1,len(models),figsize=(60,18))

for i in range(len(models)):
        axs[i].tick_params(axis='both', which='major', labelsize=50)
        axs[i].set_title('Model: %s'% (modelnames[i]),fontdict={'family':'Times New Roman','size':60})
        axs[i].set_xlabel(r'$X$', fontdict={'family': 'Times New Roman', 'size': 40})
        axs[i].set_ylabel(r'$Y$', fontdict={'family': 'Times New Roman', 'size': 40})
        axs[i].set_xlim(0, 1)
        axs[i].set_ylim(ylims[i])
        axs[i].plot(x0, quant0[i], alpha=0.9,lw=4)
        axs[i].scatter(x[i], y[i], color = "k", alpha=0.3,label='Data',s=32)
        axs[i].legend(names,loc=positions[i],fontsize=30)
        
#%%
from generate import gen_univ
from generate import quant_univ
import torch
import matplotlib.pyplot as plt

# Set parameters
SIZE = 2**9
taus = torch.Tensor([0.05, 0.25, 0.5, 0.75, 0.95]).unsqueeze(1)
models = ['linear', 'wave', 'angle']
errors = ['t', 'expx_t', 'sinex_t']
modelnames = ['Linear', 'Wave', 'Angle']
errornames = [r'$t(1)$', r'Exp-$t(2)$', r'Sine-$t(2)$']
x0 = torch.linspace(0, 1, 1000).unsqueeze(dim=1)

# Prepare data containers
x = []
y = []
quant0 = []

# Generate data for each model
for j in range(len(errors)):
    dataset = gen_univ(size=SIZE, model=models[j], error=errors[j], df=2, sigma=1)
    x.append(dataset[:][0].data.numpy())
    y.append(dataset[:][1].data.numpy())
    quant0.append(quant_univ(x0, taus, model=models[j], error=errors[j], df=2, sigma=1))

ylims = [[-4, 7], [-7, 7], [0, 7.5]]
positions = ['upper left', 'lower left', 'upper left']

# Define line styles and colors for quantile curves
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]  # 5 distinguishable styles
colors = ['deepskyblue','darkorange', 'lime', 'r', 'darkviolet' ]  # Original colors
names = (r'$\tau=0.05$', r'$\tau=0.25$', r'$\tau=0.5$', r'$\tau=0.75$', r'$\tau=0.95$', 'Data')

# Plot and show the ground truth models and their conditional quantile curves
figs, axs = plt.subplots(1, len(models), figsize=(60, 18))

for i in range(len(models)):
    axs[i].tick_params(axis='both', which='major', labelsize=50)
    axs[i].set_title('Model: %s' % (modelnames[i]), fontdict={'family': 'Times New Roman', 'size': 60})
    axs[i].set_xlabel(r'$X$', fontdict={'family': 'Times New Roman', 'size': 40})
    axs[i].set_ylabel(r'$Y$', fontdict={'family': 'Times New Roman', 'size': 40})
    axs[i].set_xlim(0, 1)
    axs[i].set_ylim(ylims[i])
    # Store handles for legend
    handles = []
    labels = []
    # Plot each quantile curve and save handle
    for k in range(5):
        line, = axs[i].plot(
            x0, quant0[i][:, k],
            alpha=0.9, lw=7,
            color=colors[k],
            linestyle=linestyles[k]
        )
        handles.append(line)
        labels.append(names[k])
    # Plot the data points and save handle
    scatter = axs[i].scatter(x[i], y[i], color="k", alpha=0.3, s=32)
    handles.append(scatter)
    labels.append(names[-1])
    # Show legend with correct handles and labels
    axs[i].legend(handles, labels, loc=positions[i], fontsize=32)

plt.savefig("figure3.png", dpi=400)
plt.show()
