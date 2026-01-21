import pandas as pd, numpy as np, os, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv("ai4i2020.csv", encoding="utf-8")

# 파생변수 만들기 power랑 온도차이
df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power"] = (2*np.pi/60) * df["Rotational speed [rpm]"] * df["Torque [Nm]"]

# osf용 strain 변수 만들기 
df["Variant"] = df["Type"]
df["strain"] = df["Tool wear [min]"] * df["Torque [Nm]"]  

labels = ["TWF","HDF","PWF","OSF","RNF"]

def plot_type_distribution(label_col):
    # Type(L/M/H) 분포: failure랑 non-failure
    fail = df[df[label_col] == 1]
    ok = df[df[label_col] == 0]
    cats = ["L","M","H"] # 카테고리

    fail_counts = fail["Type"].value_counts().reindex(cats).fillna(0)
    ok_counts = ok["Type"].value_counts().reindex(cats).fillna(0)

    x = np.arange(len(cats))
    width = 0.38 # 이게 젤 깔끔

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, ok_counts/len(ok), width, label="non-failure share")
    plt.bar(x + width/2, fail_counts/len(fail) if len(fail)>0 else 0, width, label="failure share")
    plt.xticks(x, cats)
    plt.ylabel("Proportion within group")
    plt.title(f"{label_col}: Type distribution (failure vs non-failure)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def plot_box(label_col, feature):
        # 피쳐 박스플롯: failure랑 non-failure
        fail = df[df[label_col] == 1][feature].astype(float)
        ok = df[df[label_col] == 0][feature].astype(float)

        plt.figure(figsize=(6,4))
        plt.boxplot([ok, fail], labels=["non-failure", "failure"], showfliers=False)
        plt.ylabel(feature)
        plt.title(f"{label_col}: {feature} (boxplot)")
        plt.tight_layout()
        plt.show()

    def plot_hist(label_col, feature, bins=35, density=True, logy=False):
        """feature 히스토그램: failure vs non-failure"""
        fail = df[df[label_col] == 1][feature].astype(float)
        ok = df[df[label_col] == 0][feature].astype(float)

        plt.figure(figsize=(6,4))
        plt.hist(ok, bins=bins, alpha=0.6, label="non-failure", density=density)
        plt.hist(fail, bins=bins, alpha=0.6, label="failure", density=density)
        plt.xlabel(feature)
        plt.ylabel("Density" if density else "Count")
        plt.title(f"{label_col}: {feature} (hist)")
        if logy:
            plt.yscale("log")
            plt.ylabel(("Density" if density else "Count") + " (log scale)")
            plt.title(f"{label_col}: {feature} (hist, log y)")
        plt.legend()
        plt.tight_layout()
        plt.show()