
###################################################################################################

import json
from glob import glob

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

import seaborn as sns
sns.set_style("darkgrid")

import matplotlib.pyplot as plt

###################################################################################################

_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

_IMSHOW_OPTS = dict(cmap="Greys_r", interpolation="bilinear", vmin=0, vmax=1)

###################################################################################################

def visualize_samples(X, y, samples=15):

    y = y.argmax(axis=1)
    img_h, img_w = X.shape[1:3]
    samples = max(5, min(15, samples))
    
    fig, axs = plt.subplots(10, samples, figsize=(samples, 10))

    for i in range(10):

        choices = np.random.choice(np.where(y==i)[0], samples, replace=False)

        for j, choice in enumerate(choices):

            axs[i, j].axis("off")
            axs[i, j].imshow(X[choice, :, :, 0], **_IMSHOW_OPTS)

        axs[i, -1].text(
            1.2 * img_w, 0.5 * img_h,
            f"({i}) {_LABELS[i]}",
            va="center", fontsize=14)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    
def visualize_augmentations(augmenter, X_sample, y_sample):

    samples = X_sample.shape[0]
    generator = augmenter.flow(X_sample, y_sample, samples, shuffle=False)

    X_aug, y_aug = next(generator)

    fig, axs = plt.subplots(2, samples, figsize=(14, 3))

    for i in range(X_sample.shape[0]):

        axs[0, i].axis("off")
        axs[0, i].imshow(X_sample[i, :, :, 0], **_IMSHOW_OPTS)

        axs[1, i].axis("off")
        axs[1, i].imshow(X_aug[i, :, :, 0], **_IMSHOW_OPTS)

    axs[0, -1].text(
        30, 14, "Original\ndata batch",
        va="center", fontsize=14)

    axs[1, -1].text(
        30, 14, "Augmented\ndata batch",
        va="center", fontsize=14)

    fig.tight_layout()


def visualize_schedulers(schd_1, epochs_1, schd_2, epochs_2):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    
    x1 = np.arange(epochs_1)
    y1 = [schd_1(xi) for xi in x1]
    
    x2 = np.arange(epochs_2)
    y2 = [schd_2(xi) for xi in x2]
    
    ax1.plot(x1+1, y1)
    ax2.plot(x2+1, y2)
    
    for i, ax in enumerate((ax1, ax2)):
        
        ax.set_ylim((0, 0.0012))
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_title(f"Scheduler {i+1}", fontsize=14)


def plot_learning_curves(model_type, metric, skip_epochs=0, results_dir="results"):
    
    tables = []

    for results_file in glob(f"{results_dir}/*.json"):

        if model_type not in results_file:
            continue
        
        with open(results_file, "r") as file:
            result = json.load(file)

        if metric == "error":
            metric = "accuracy"
            
        y_train = 1 - np.array(result[metric])
        y_val = 1 - np.array(result[f"val_{metric}"])
        
        y_train = y_train[skip_epochs:]
        y_val = y_val[skip_epochs:]
        
        if metric == "loss":
            y_train = 1 - y_train
            y_val = 1 - y_val

        x = 1 + np.arange(len(y_train))

        model = result["model_name"]
        
        pos = results_file.find(model)

        aug = results_file[pos:]
        aug = aug.replace(f"{model}_", "")
        aug = aug.replace(".json", "").replace("_", " + ")

        tables.append(pd.DataFrame({
            "model": [model] * len(x), "aug": [aug] * len(x),
            "split": ["training"] * len(x), "x": x, "y": y_train}))

        tables.append(pd.DataFrame({
            "model": [model] * len(x), "aug": [aug] * len(x),
            "split": ["validation"] * len(x), "x": x, "y": y_val}))

    plot_data = pd.concat(tables)
    
    with sns.plotting_context("notebook", font_scale=1.2):
    
        plot = sns.relplot(
            kind="line", x="x", y="y",
            row="model", col="aug",
            hue="split", data=plot_data)

        for ax in plot.axes.flatten():
            ax.set_xlabel("")
            ax.set_ylabel("")


def plot_confusion_matrix(y_true, y_pred):
    
    conf_mat = confusion_matrix(y_true, y_pred)

    total = conf_mat.sum()

    recalls = 100 * np.diag(conf_mat) / conf_mat.sum(axis=1)
    precisions = 100 * np.diag(conf_mat) / conf_mat.sum(axis=0)

    annot = np.array([
        [f"{e}\n{100 * e / total:.2f}%" for e in row]
        for row in conf_mat])

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(
        data=np.log(1 + conf_mat), ax=ax,
        cbar=False, cmap="Blues",
        linewidths=1, linecolor="#BBBBBB",
        annot=annot, annot_kws=dict(fontsize=11, fontweight="bold"), fmt="s")

    ax.set_ylim((10, 0))

    ax.set_yticklabels(_LABELS, rotation=0, fontsize=14)
    ax.set_xticklabels(_LABELS, rotation=30, fontsize=14)

    for i, r in enumerate(recalls):
        ax.text(10.2, i + 0.5, f"{r:.2f}%", va="center", fontsize=12)

    for i, p in enumerate(precisions):
        ax.text(i + 0.5, -0.2, f"{p:.2f}%", ha="center", fontsize=12)

    ax.set_ylabel("True class (Recall)", fontsize=16)
    ax.set_xlabel("Predicted class (Precision)", fontsize=16)


def show_demo(X_test, y_test, model):

    i = np.random.randint(len(X_test), size=1)

    x = X_test[i]
    y = y_test[i].argmax()

    probs = model.predict(x)[0]
    probs_data = pd.DataFrame(dict(x=np.arange(10), y=probs))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4),
        gridspec_kw={"width_ratios": [1, 2]})

    # plot 1

    ax1.axis("off")
    ax1.set_title(_LABELS[y], fontsize=16)
    ax1.imshow(x[0, :, :, 0], **_IMSHOW_OPTS)

    # plot 2

    probs_data.plot("x", "y", kind="barh", width=.8, ax=ax2)

    ax2.invert_yaxis()
    ax2.legend().remove()
    ax2.set_xlim((0, 1.35))

    ax2.get_yaxis().set_visible(False)
    ax2.set_xticks(np.linspace(0, 1, 11)[1:])
    ax2.tick_params(axis="x", which="major", labelsize=12)

    for j, (prob, label) in enumerate(zip(probs, _LABELS)):

        text_opts=dict(va="center",
                       fontsize=12,
                       fontfamily="Consolas")

        if _LABELS[y] == label:
            text_opts["fontweight"] = "bold"

        ax2.text(prob, j, f" {100 * prob:.2f}% {label}", **text_opts)

    fig.tight_layout()
    
###################################################################################################

