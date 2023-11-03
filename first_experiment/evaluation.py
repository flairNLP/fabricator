import glob
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_scores() -> pd.DataFrame:
    path = "/glusterfs/dfs-gfs-dist/goldejon/initial-starting-point-generation"
    files = glob.glob(path + "/*/results.json")

    scores = {
        "Dataset Size": [],
        "Dataset Name": [],
        "Initialization Strategy": [],
        "Accuracy": [],
    }

    for file in files:
        with open(file, "r") as f:
            res = json.load(f)

        infos = file.split("/")[-2].split("_")

        if len(infos) == 4:
            model, dataset, dataset_size, init_strategy = infos
            embedding_model = None
        elif len(infos) == 5:
            model, dataset, dataset_size, init_strategy, embedding_model = infos
        else:
            raise ValueError("Wrong number of infos.")

        scores["Dataset Size"].append(int(dataset_size))
        scores["Dataset Name"].append(dataset)
        scores["Initialization Strategy"].append(f"{init_strategy} ({embedding_model})")
        scores["Accuracy"].append(res["test_accuracy"])

    return pd.DataFrame(data=scores)

if __name__ == "__main__":
    scores = load_scores()
    full_finetuning = scores[scores["Dataset Size"] == 0]
    scores = scores[scores["Dataset Size"] > 0]

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)

    g = sns.FacetGrid(scores, col="Dataset Name", sharey=False, height=4, aspect=1.5)
    g.map(sns.lineplot, "Dataset Size", "Accuracy", "Initialization Strategy", ci=None, marker="o")
    g.set(xscale="log")

    for idx, col in enumerate(g.col_names):
        ft_score = full_finetuning[full_finetuning["Dataset Name"] == col]["Accuracy"].iloc[0]
        g.axes[0][idx].axhline(ft_score, ls="--", color="black")

    g.add_legend()

    plt.show()

