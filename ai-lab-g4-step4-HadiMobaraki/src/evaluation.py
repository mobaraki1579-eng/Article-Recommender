# %% [markdown]
# # گام ۴: ارزیابی اولیه (داده و ارزیابی)
# **نویسنده:** هادی مبارکی  |  **نقش:** داده و ارزیابی  |  **درس:** آزمایشگاه هوش مصنوعی
#
# **مسئله:** دسته‌بندی چکیده مقالات فارسی در ۷ دسته (برچسب ۱ تا ۷).
# **کار من در این گام:** اجرای ارزیابی اولیه؛ یعنی تقسیم داده، baseline، معیارها (دستی و با scikit-learn)، مقایسه چند مدل ساده و نوشتن محدودیت‌ها.
#
# اجرا: `python src/evaluation.py` یا نوت‌بوک `notebooks/step4_evaluation.ipynb` (سلول‌ها را از بالا به پایین اجرا کنید).

# %% [markdown]
# ## سلول ۱: داده و بررسی اولیه

# %%
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "corpus.json")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "corpus.json"          # اگر فایل کنار اسکریپت بود
OUT = "results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

with open(DATA_PATH, encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

df["text"] = df["title"] + " " + df["abstract"]       # عنوان و چکیده را با هم استفاده می‌کنیم
df["n_words"] = df["text"].str.split().str.len()

print("shape:", df.shape)
print("missing values:", int(df.isna().sum().sum()))
print("duplicate titles:", int(df["title"].duplicated().sum()),
      "| duplicate abstracts:", int(df["abstract"].duplicated().sum()))
counts = df["label"].value_counts().sort_index()
print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(1)}))
print(df.groupby("label")["n_words"].agg(["mean", "median", "std"]).round(1))

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(counts.index.astype(str), counts.values, color="#4c72b0")
for i, v in zip(counts.index.astype(str), counts.values):
    ax[0].text(i, v + 40, str(v), ha="center", fontsize=9)
ax[0].set_xlabel("label"); ax[0].set_ylabel("number of papers"); ax[0].set_title("Class distribution")
df.boxplot(column="n_words", by="label", ax=ax[1], showfliers=False)
ax[1].set_xlabel("label"); ax[1].set_ylabel("words (title + abstract)"); ax[1].set_title("Text length per class")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "01_class_distribution.png"), dpi=150)
plt.close()

# %% [markdown]
# ## سلول ۲: تقسیم آموزش/آزمون (دستی) و baseline
# چون ۷ کلاس داریم و کلاس‌ها نامتوازن‌اند، تقسیم را برای هر کلاس جداگانه انجام می‌دهم (۷۰٪ آموزش، ۳۰٪ آزمون) تا نسبت کلاس‌ها در هر دو بخش یکسان بماند.

# %%
rng = np.random.default_rng(0)
labels = np.sort(df["label"].unique())
train_idx, test_idx = [], []
for c in labels:
    idx = np.where(df["label"].to_numpy() == c)[0]
    idx = rng.permutation(idx)
    n_tr = int(0.7 * len(idx))
    train_idx.extend(idx[:n_tr])
    test_idx.extend(idx[n_tr:])
train_idx = rng.permutation(np.array(train_idx))
test_idx = rng.permutation(np.array(test_idx))

train, test = df.iloc[train_idx], df.iloc[test_idx]
y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
print("train rows:", len(train), " test rows:", len(test))
print("test class share:", (pd.Series(y_test).value_counts(normalize=True).sort_index() * 100).round(1).to_dict())

# baseline: همیشه پرتکرارترین کلاس آموزش را بگو
majority = pd.Series(y_train).value_counts().idxmax()
base_pred = np.full(len(y_test), majority)
print("majority class:", majority)
print("baseline accuracy:", round((base_pred == y_test).mean(), 3))

# %% [markdown]
# ## سلول ۳: معیارها با دست
# برای چندکلاسه از ماتریس درهم‌ریختگی شروع می‌کنم؛ precision و recall و F1 را برای هر کلاس حساب می‌کنم و بعد میانگین (macro و weighted) می‌گیرم.

# %%
def confusion(y_true, y_pred, classes):
    pos = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[pos[t], pos[p]] += 1               # سطر = برچسب واقعی، ستون = پیش‌بینی
    return m


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "macro_precision": precision.mean(), "macro_recall": recall.mean(), "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
    }


cm_base = confusion(y_test, base_pred, labels)
m_base = metrics_from_cm(cm_base)
print("baseline (by hand): accuracy=%.3f  macro_recall=%.3f  macro_f1=%.3f  weighted_f1=%.3f" % (
    m_base["accuracy"], m_base["macro_recall"], m_base["macro_f1"], m_base["weighted_f1"]))
print("per-class recall:", m_base["recall"].round(3))

# %% [markdown]
# ## سلول ۴: k-NN از صفر روی TF-IDF
# متن را به بردار TF-IDF تبدیل می‌کنم. **مهم:** vectorizer فقط روی داده آموزش fit می‌شود تا اطلاعات آزمون نشت نکند. فاصله: شباهت کسینوسی (چون بردارها نرمال‌اند، ضرب داخلی همان شباهت است).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 1), sublinear_tf=True, min_df=2)
X_train = vec.fit_transform(train["text"])
X_test = vec.transform(test["text"])
print("TF-IDF shape (train):", X_train.shape)


def knn_predict_all(Xtr, ytr, Xte, k=5, chunk=500):
    preds = []
    for s in range(0, Xte.shape[0], chunk):
        sim = (Xte[s:s + chunk] @ Xtr.T).toarray()                  # شباهت با همه نمونه‌های آموزش
        nearest = np.argpartition(-sim, k - 1, axis=1)[:, :k]       # k نزدیک‌ترین
        for row, nn in zip(sim, nearest):
            votes = np.bincount(ytr[nn], minlength=ytr.max() + 1)
            # اگر آرا مساوی شد، همسایه‌ای که شبیه‌تر است حساب می‌شود
            weights = np.bincount(ytr[nn], weights=row[nn], minlength=ytr.max() + 1)
            best = np.lexsort((weights, votes))[-1]
            preds.append(best)
    return np.array(preds)


knn_pred = knn_predict_all(X_train, y_train, X_test, k=5)
cm_knn = confusion(y_test, knn_pred, labels)
m_knn = metrics_from_cm(cm_knn)
print("k-NN (from scratch, k=5): accuracy=%.3f  macro_precision=%.3f  macro_recall=%.3f  macro_f1=%.3f" % (
    m_knn["accuracy"], m_knn["macro_precision"], m_knn["macro_recall"], m_knn["macro_f1"]))

# %% [markdown]
# # قسمت ۲: همین ارزیابی با scikit-learn
# ## سلول ۵: چند مدل، همان معیارها
# اول بررسی می‌کنم که معیارهای دستی من با scikit-learn یکی باشند، بعد مدل‌ها را مقایسه می‌کنم.

# %%
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)

# اعتبارسنجی کد دستی با scikit-learn
assert abs(accuracy_score(y_test, knn_pred) - m_knn["accuracy"]) < 1e-9
assert abs(f1_score(y_test, knn_pred, average="macro") - m_knn["macro_f1"]) < 1e-9
assert abs(f1_score(y_test, base_pred, average="macro", zero_division=0) - m_base["macro_f1"]) < 1e-9
assert (confusion_matrix(y_test, knn_pred, labels=labels) == cm_knn).all()
print("hand-made metrics match scikit-learn")

models = {
    "baseline (always majority)": DummyClassifier(strategy="most_frequent"),
    "random (class prior)": DummyClassifier(strategy="stratified", random_state=0),
    "k-NN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    "naive Bayes": MultinomialNB(alpha=0.1),
    "logistic regression": LogisticRegression(max_iter=1000, C=10),
}
fitted, preds, rows = {}, {}, []
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    fitted[name], preds[name] = model, p
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, p),
        "macro_precision": precision_score(y_test, p, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, p, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, p, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, p, average="weighted", zero_division=0),
    })
table1 = pd.DataFrame(rows).round(3)
print(table1.to_string(index=False))
table1.to_csv(os.path.join(OUT, "metrics_summary.csv"), index=False)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(table1))
ax.bar(x - 0.2, table1["accuracy"], 0.4, label="accuracy", color="#9aa5b1")
ax.bar(x + 0.2, table1["macro_f1"], 0.4, label="macro F1", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(table1["model"], rotation=20, ha="right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy vs macro F1 (test set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "02_model_comparison.png"), dpi=150)
plt.close()

# %% [markdown]
# ## سلول ۶: گزارش هر کلاس و ماتریس درهم‌ریختگی
# بهترین مدل را بر اساس macro F1 انتخاب می‌کنم (نه دقت)، چون کلاس‌ها نامتوازن‌اند.

# %%
best_name = table1.sort_values("macro_f1", ascending=False).iloc[0]["model"]
print("best model by macro F1:", best_name)
rep = classification_report(y_test, preds[best_name], labels=labels, digits=3, zero_division=0, output_dict=True)
per_class = pd.DataFrame({str(c): rep[str(c)] for c in labels}).T.round(3)
per_class["support"] = per_class["support"].astype(int)
print(per_class)
per_class.to_csv(os.path.join(OUT, "per_class_best_model.csv"))

base_rep = classification_report(y_test, base_pred, labels=labels, digits=3, zero_division=0, output_dict=True)
pd.DataFrame({str(c): base_rep[str(c)] for c in labels}).T.round(3).to_csv(os.path.join(OUT, "per_class_baseline.csv"))

cm_best = confusion_matrix(y_test, preds[best_name], labels=labels)
cm_norm = cm_best / cm_best.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, mat, title, fmt in [(ax[0], cm_best, "counts", "d"), (ax[1], cm_norm, "row-normalized (recall)", ".2f")]:
    a.imshow(mat, cmap="Blues")
    a.set_xticks(range(len(labels))); a.set_yticks(range(len(labels)))
    a.set_xticklabels(labels); a.set_yticklabels(labels)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"{best_name} - {title}", fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            a.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8,
                   color="white" if val > mat.max() * 0.55 else "black")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "03_confusion_matrix_best.png"), dpi=150)
plt.close()

# %%
# بیشترین اشتباه‌ها: کدام دو کلاس بیشتر با هم قاطی می‌شوند؟
pairs = []
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        if i != j and cm_best[i, j] > 0:
            pairs.append({"true": a, "predicted": b, "count": int(cm_best[i, j]),
                          "share_of_true_class": round(cm_best[i, j] / cm_best[i].sum(), 3)})
errors = pd.DataFrame(pairs).sort_values("count", ascending=False).head(8)
print(errors.to_string(index=False))
errors.to_csv(os.path.join(OUT, "top_confusions.csv"), index=False)

# %% [markdown]
# ## سلول ۷: منحنی بیش‌برازش (k-NN)

# %%
curve = []
for k in [1, 3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute").fit(X_train, y_train)
    curve.append({"k": k,
                  "train_acc": knn.score(X_train, y_train),
                  "test_acc": knn.score(X_test, y_test),
                  "test_macro_f1": f1_score(y_test, knn.predict(X_test), average="macro")})
curve = pd.DataFrame(curve).round(3)
print(curve)
curve.to_csv(os.path.join(OUT, "knn_k_curve.csv"), index=False)

ax = curve.plot(x="k", y=["train_acc", "test_acc", "test_macro_f1"], marker="o", figsize=(6.5, 4))
ax.set_ylabel("score"); ax.set_title("k-NN: train vs test")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "04_knn_k_curve.png"), dpi=150)
plt.close()

# %% [markdown]
# ## سلول ۸: اعتبارسنجی ۵-تکه‌ای
# فقط یک تقسیم ممکن است شانسی باشد، پس macro F1 را روی ۵ تکه هم حساب می‌کنم. vectorizer داخل pipeline است تا در هر تکه فقط روی داده آموزش همان تکه fit شود.

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_rows = []
for name, model in models.items():
    pipe = make_pipeline(TfidfVectorizer(max_features=50000, sublinear_tf=True, min_df=2), model)
    scores = cross_val_score(pipe, df["text"], df["label"], cv=skf, scoring="f1_macro")
    cv_rows.append({"model": name, "cv_macro_f1_mean": scores.mean(), "cv_macro_f1_std": scores.std()})
cv_table = pd.DataFrame(cv_rows).round(3)
print(cv_table.to_string(index=False))
cv_table.to_csv(os.path.join(OUT, "cv_macro_f1.csv"), index=False)

# %% [markdown]
# ## سلول ۹: ذخیره نتایج

# %%
summary = {
    "n_rows": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test)),
    "classes": [int(c) for c in labels],
    "majority_class": int(majority),
    "baseline_accuracy": round(float(m_base["accuracy"]), 4),
    "baseline_macro_f1": round(float(m_base["macro_f1"]), 4),
    "best_model": best_name,
    "best_accuracy": float(table1.set_index("model").loc[best_name, "accuracy"]),
    "best_macro_f1": float(table1.set_index("model").loc[best_name, "macro_f1"]),
    "split": "stratified 70/30, rng seed 0",
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
