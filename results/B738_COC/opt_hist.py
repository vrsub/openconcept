import os
import argparse
import matplotlib.pyplot as plt
from numpy import size
from pyoptsparse import History

plt.rcParams["text.usetex"] = True  # Comment out if latex installation is not present
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["lines.linewidth"] = 2.0

parser = argparse.ArgumentParser()
parser.add_argument("--histFile", type=str, default="opt.hst")
parser.add_argument("--outputDir", type=str, default="./")
args = parser.parse_args()

optHist = History(args.histFile)
histVals = optHist.getValues(major='True')
print(histVals.keys())
# print(histVals)

plt.figure(figsize=(8, 6))
plt.plot(histVals['COC.COC.COC'], '-o')
plt.ylabel('Obj: COC [\$]')
plt.xlabel('Major Iteration Number')
# fig, axes = plt.subplots(nrows=6, sharex=True, constrained_layout=True, figsize=(14, 10))

# axes[0].plot(histVals['analysis.descent.acmodel.intfuel.fuel_used_final'], label="Objective")
# axes[0].set_xlabel('Iteration', fontsize=16)
# axes[0].set_ylabel("Objective \n(Fuel Burn)", rotation="horizontal", ha="right")

# axes[2].plot("nMajor", "alpha_fc", data=histValues, label="Alpha")
# axes[2].set_ylabel(r"$\alpha [^\circ]$", rotation="horizontal", ha="right", fontsize=24)

# axes[3].plot("nMajor", "twist", data=histValues, label="Twist")
# axes[3].set_ylabel(r"Twist $[^\circ]$", rotation="horizontal", ha="right", fontsize=24)

# axes[4].plot("nMajor", "cl_con_fc", data=histValues, label="cl_con")
# axes[4].set_ylabel("Lift constraint", rotation="horizontal", ha="right", fontsize=24)

# for ax in axes:
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)

#     # Drop the rest of the spines
#     ax.spines["left"].set_position(("outward", 12))
#     ax.spines["bottom"].set_position(("outward", 12))

plt.savefig(os.path.join(args.outputDir, "737_COC_obj.png"))