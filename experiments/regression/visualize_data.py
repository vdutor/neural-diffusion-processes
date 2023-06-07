#%%
import numpy as np
import matplotlib.pyplot as plt


DATASETS = [
    "se",
    "matern",
    "sawtooth",
    "step",
]

TASKS = [
    "training",
    "interpolation",
]


def plot_data(x_target, y_target, ax, context=None,):
    ax.plot(x_target, y_target, "C0.", label="target")
    if context is not None:
        ax.plot(context[0], context[1], "C3.", label="context")


ns = 10
fig, axes = plt.subplots(len(DATASETS), len(TASKS), figsize=(8, 8))
for i, dataset in enumerate(DATASETS):
    for j, task in enumerate(TASKS):
        data = np.load(f"data/{dataset}_1_{task}.npz")
        xs = data["x_target"][:ns][..., 0].T
        ys = data["y_target"][:ns][..., 0].T
        xc = data["x_context"][:ns][..., 0].T
        yc = data["y_context"][:ns][..., 0].T

        plot_data(xs, ys, axes[i, j], context=(xc, yc))

        if j == 0:
            axes[i, j].set_ylabel(dataset)
        if i == 0:
            axes[i, j].set_title(task)
        if i == 0 and j == 0:
            axes[i, j].legend()

# %%
data.keys()
# %%
data2 = {
    k: v.astype(np.float32) for k, v in data.items()
}
# %%
