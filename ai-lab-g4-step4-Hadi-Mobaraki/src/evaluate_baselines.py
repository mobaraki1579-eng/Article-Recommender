"""
Step 4 - Data & Evaluation task (AI Lab, group 4)
Choose baseline + metric, run the first evaluation, record results and limitations.

Run from the repo root:
    python src/evaluate_baselines.py --corpus data/corpus.json
Outputs go to results/.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix,
                             ConfusionMatrixDisplay)

SEED = 0

# ---------------------------------------------------------------- text cleaning
_AR = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا"})
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize(text: str) -> str:
    text = text.translate(_AR).translate(_FA_DIGITS)
    text = text.replace("\u200c", " ")                    # zero-width non-joiner -> space
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)     # harakat + tatweel
    text = re.sub(r"[^\w\s]", " ", text)                  # punctuation
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------- metrics by hand
def metrics_by_hand(y_true, y_pred, labels):
    """Multiclass confusion matrix + per-class P/R/F1 + macro-F1, NumPy only."""
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {"cm": cm, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": tp.sum() / cm.sum(), "macro_f1": f1.mean(), "macro_recall": rec.mean()}


def summarize(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0)}


def tfidf(**kw):
    base = dict(preprocessor=normalize, ngram_range=(1, 2), min_df=3, max_df=0.9,
                sublinear_tf=True, max_features=150_000)
    base.update(kw)
    return TfidfVectorizer(**base)


def build_models():
    return {
        "B1 majority class":              make_pipeline(DummyClassifier(strategy="most_frequent")),
        "B2 random (class priors)":       make_pipeline(DummyClassifier(strategy="stratified", random_state=SEED)),
        "B3 random (uniform)":            make_pipeline(DummyClassifier(strategy="uniform", random_state=SEED)),
        "M1 TF-IDF + ComplementNB":       make_pipeline(tfidf(), ComplementNB()),
        "M2 TF-IDF + LogisticRegression": make_pipeline(tfidf(), LogisticRegression(max_iter=2000, C=10)),
        "M3 TF-IDF + LinearSVC":          make_pipeline(tfidf(), LinearSVC(C=0.5, random_state=SEED)),
    }


# ---------------------------------------------------------------- main
def main(corpus_path: str, out: str = "results"):
    out = Path(out)
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(json.load(open(corpus_path, encoding="utf-8")))
    df["text"] = df["title"].str.strip() + " " + df["abstract"].str.strip()
    labels = sorted(df["label"].unique())

    # ---- 1. data audit -------------------------------------------------------
    words = df["abstract"].str.split().str.len()
    dup_title = df.duplicated("title", keep=False)
    vc = df["label"].value_counts()
    audit = {
        "n_records": int(len(df)), "n_classes": len(labels),
        "class_counts": {int(k): int(v) for k, v in vc.sort_index().items()},
        "class_share_pct": {int(k): round(100 * v / len(df), 2) for k, v in vc.sort_index().items()},
        "imbalance_ratio_max_over_min": round(vc.max() / vc.min(), 2),
        "empty_abstracts": int((df["abstract"].str.strip() == "").sum()),
        "duplicate_abstracts": int(df.duplicated("abstract").sum()),
        "records_sharing_a_title": int(dup_title.sum()),
        "duplicate_titles_with_conflicting_label": int((df[dup_title].groupby("title")["label"].nunique() > 1).sum()),
        "abstract_words": {"min": int(words.min()), "median": float(words.median()),
                           "mean": round(float(words.mean()), 1), "max": int(words.max())},
        "abstracts_under_20_words": int((words < 20).sum()),
        "abstracts_with_latin_words_pct": round(100 * df["abstract"].str.contains(r"[A-Za-z]{3,}").mean(), 1),
    }
    json.dump(audit, open(out / "data_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    sv = vc.sort_index()
    plt.figure(figsize=(6, 3.5))
    plt.bar(sv.index.astype(str), sv.values, color="#4f46e5")
    for i, v in enumerate(sv.values):
        plt.text(i, v + 60, f"{100 * v / len(df):.1f}%", ha="center", fontsize=8)
    plt.xlabel("class (label)")
    plt.ylabel("records")
    plt.title(f"Class distribution ({len(df):,} records)")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150)
    plt.close()

    # ---- 2. split (same recipe as lab4: 70/30, stratified, seed 0) ------------
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.3,
                                              random_state=SEED, stratify=df["label"])
    json.dump({"train": int(len(X_tr)), "test": int(len(X_te)),
               "test_class_counts": {int(k): int(v) for k, v in y_te.value_counts().sort_index().items()}},
              open(out / "split.json", "w"), indent=2)

    # ---- 3. models --------------------------------------------------------------
    models = build_models()
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds[name] = m.predict(X_te)
        rows.append({"model": name, **summarize(y_te, preds[name])})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(out / "holdout_results.csv", index=False)

    # ---- 4. by-hand check (like lab4 part 1) --------------------------------------
    best = table.loc[table["macro_F1"].idxmax(), "model"]
    hand = metrics_by_hand(y_te.to_numpy(), preds[best], labels)
    sk_f1 = f1_score(y_te, preds[best], average="macro")
    assert abs(hand["macro_f1"] - sk_f1) < 1e-9, "hand-made macro-F1 differs from sklearn"
    assert abs(hand["accuracy"] - accuracy_score(y_te, preds[best])) < 1e-12

    # ---- 5. per-class report -------------------------------------------------------
    def per_class(p):
        pr, rc, f1, sup = precision_recall_fscore_support(y_te, p, labels=labels, zero_division=0)
        return pd.DataFrame({"class": labels, "precision": pr, "recall": rc, "f1": f1, "support": sup}).round(3)

    pc_best, pc_base = per_class(preds[best]), per_class(preds["B1 majority class"])
    pc_best.to_csv(out / "per_class_best_model.csv", index=False)
    pc_base.to_csv(out / "per_class_majority_baseline.csv", index=False)

    for name, cmap, fname in [(best, "Blues", "confusion_matrix_best.png"),
                              ("B1 majority class", "Oranges", "confusion_matrix_baseline.png")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_te, preds[name], labels=labels), display_labels=labels).plot(
            ax=ax, cmap=cmap, colorbar=False, values_format="d")
        ax.set_title(f"{name} (test set, n={len(y_te)})", fontsize=9)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=150)
        plt.close()

    ax = table.set_index("model")[["accuracy", "macro_F1"]].plot.barh(figsize=(7, 4), color=["#9ca3af", "#4f46e5"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("score (test set)")
    ax.invert_yaxis()
    plt.title("Accuracy vs macro-F1")
    plt.tight_layout()
    plt.savefig(out / "model_comparison.png", dpi=150)
    plt.close()

    # ---- 6. robustness: 5-fold stratified CV on the full data -----------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name in ["B1 majority class", "B2 random (class priors)", "M1 TF-IDF + ComplementNB",
                 "M2 TF-IDF + LogisticRegression", "M3 TF-IDF + LinearSVC"]:
        accs, f1s = [], []
        for tr, te in skf.split(df["text"], df["label"]):
            m = clone(models[name]).fit(df["text"].iloc[tr], df["label"].iloc[tr])
            p = m.predict(df["text"].iloc[te])
            accs.append(accuracy_score(df["label"].iloc[te], p))
            f1s.append(f1_score(df["label"].iloc[te], p, average="macro", zero_division=0))
        cv_rows.append({"model": name, "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                        "macro_F1_mean": np.mean(f1s), "macro_F1_std": np.std(f1s)})
    pd.DataFrame(cv_rows).round(3).to_csv(out / "cv5_results.csv", index=False)

    # ---- 7. which input field matters? (limitation probe) --------------------------
    abl = []
    for field in ["title", "abstract", "text"]:
        m = clone(models["M3 TF-IDF + LinearSVC"]).fit(df[field].loc[X_tr.index], y_tr)
        abl.append({"input": "title+abstract" if field == "text" else field,
                    **summarize(y_te, m.predict(df[field].loc[X_te.index]))})
    pd.DataFrame(abl).round(3).to_csv(out / "input_ablation.csv", index=False)

    # ---- 8. summary -----------------------------------------------------------------
    summary = {"primary_metric": "macro_F1", "best_model": best,
               "majority_baseline": table[table.model == "B1 majority class"].iloc[0].to_dict(),
               "best": table[table.model == best].iloc[0].to_dict(),
               "hand_vs_sklearn_macro_f1": [float(hand["macro_f1"]), float(sk_f1)],
               "weakest_classes_best_model": pc_best.sort_values("f1").head(3).to_dict("records")}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)

    print(json.dumps(audit, ensure_ascii=False, indent=1))
    print(table.to_string(index=False), "\n")
    print(pd.read_csv(out / "cv5_results.csv").to_string(index=False), "\n")
    print(pc_best.to_string(index=False), "\n")
    print(pd.read_csv(out / "input_ablation.csv").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.json")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.corpus, a.out)
