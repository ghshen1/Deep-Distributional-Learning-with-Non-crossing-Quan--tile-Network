from generate import gen_univ
from model.NQNet import NQNet
from model.DQR import DQR
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data Generation
taus = torch.linspace(0.1, 0.9, 9).unsqueeze(1)
model = 'linear'
error = "cross"
d = 1
df = 2
SIZE = 1000

# Train and save the predictions for one replication
data_train = gen_univ(model=model, size=SIZE, error=error, df=df)
x_train, y_train = data_train[:][0], data_train[:][1]

width = 128
lr = 0.001
net_DQR = DQR(quantiles=taus, non_crossing=False)
net_DQR.fit(x_train.numpy(), y_train.numpy(), width_vec=[d, width, width, width, len(taus)], lr=lr, epochs=100)

net_NQ = NQNet(quantiles=taus)
net_NQ.fit(x_train.numpy(), y_train.numpy(), width_vec=[d, width, width, width, len(taus)], lr=lr, epochs=100)

x_test = torch.linspace(-0.2, 1.2, 1000).unsqueeze(1)
preds = {}
preds['noncrossing'] = net_NQ.predict(x_test)
preds['crossing'] = net_DQR.predict(x_test)

# 2. Plot and show the estimations for crossing and non-crossing methods
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:pink','tab:grey','tab:brown']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 10)), (0, (5, 1)), (0, (3, 5, 1, 5))]
methods = ['With Crossing', 'Without Crossing']
names = ('Data', r'$\tau=0.1$', r'$\tau=0.2$', r'$\tau=0.3$', r'$\tau=0.4$', r'$\tau=0.5$', r'$\tau=0.6$', r'$\tau=0.7$', r'$\tau=0.8$', r'$\tau=0.9$')

figs, axs = plt.subplots(1, 2, figsize=(40, 17))
for m, method in enumerate(methods[:2]):
    axs[m].tick_params(axis='both', which='major', labelsize=35)
    axs[m].set_title('%s' % (methods[m]), fontdict={'family': 'Times New Roman', 'size': 55})
    axs[m].set_xlabel(r'$X$', fontdict={'family': 'Times New Roman', 'size': 40})
    axs[m].set_ylabel(r'$Y$', fontdict={'family': 'Times New Roman', 'size': 40})
    axs[m].set_xlim(-0.2, 1.2)
    axs[m].set_ylim([-0.8, 3.9])
    axs[m].scatter(x_train.data.numpy(), y_train.data.numpy(), color="k", alpha=0.25, label='Data', s=50)
    # Select the correct prediction array
    pred_arr = preds['crossing'] if m == 0 else preds['noncrossing']
    for j in range(len(taus)):
        # Use specified color and different linestyle for each line
        axs[m].plot(
            x_test.numpy().flatten(),
            pred_arr[:, j],
            color=colors[j % len(colors)],
            linestyle=linestyles[j % len(linestyles)],
            alpha=0.9,
            lw=7,
            label=names[j + 1]
        )
    axs[m].legend(loc='upper left', fontsize=32, ncol=2)


plt.savefig("figure1.png", dpi=300)

plt.show()
