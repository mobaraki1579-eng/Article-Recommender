import sys
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_PATH = "train_clean.csv"
TEST_PATH = "test_clean.csv"
QUERIES_PATH = "evaluation_queries.csv"
LOG_FILE = "eda_output.txt"

FIG_DIR = Path("figures")
TOP_K = 5
RANDOM_STATE = 42
MIN_WORDS = 20
DPI = 120

C1 = "#4f46e5"
C2 = "#ec4899"
C3 = "#f59e0b"
GRID = "#e5e7eb"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def section(title: str):
    print("\n" + "=" * 68)
    print(f" {title}")
    print("=" * 68)


def ensure_title_abstract(df: pd.DataFrame) -> pd.DataFrame:
    if "title" not in df.columns:
        parts = df["sentences"].str.split("\n\n", n=1, expand=True)
        df["title"] = parts[0].fillna("").str.strip()
        df["abstract"] = parts[1].fillna("").str.strip()
    return df


def load_data():
    for p in (TRAIN_PATH, TEST_PATH):
        if not Path(p).exists():
            sys.exit(f"❌ فایل پیدا نشد: {p}\n   ابتدا check_dataset.py را اجرا کن یا مسیر را اصلاح کن.")

    print("📥 در حال خواندن داده‌های تمیزشده ...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = ensure_title_abstract(train)
    test = ensure_title_abstract(test)

    for df in (train, test):
        df["n_words"] = df["sentences"].astype(str).str.split().str.len()
        df["n_chars"] = df["sentences"].astype(str).str.len()

    print(f"✅ train: {train.shape} | test: {test.shape}")
    return train, test


def plot_class_distribution(train, test):
    section("۱. نمودار میله‌ای — توزیع کلاس‌ها (Class Balance)")

    counts_train = train["labels"].value_counts().sort_index()
    counts_test = (
        test["labels"].value_counts().sort_index()
        .reindex(counts_train.index, fill_value=0)
    )

    imbalance = counts_train.max() / max(counts_train.min(), 1)
    print(f"بزرگ‌ترین کلاس: {counts_train.idxmax()} ({counts_train.max()} مقاله)")
    print(f"کوچک‌ترین کلاس: {counts_train.idxmin()} ({counts_train.min()} مقاله)")
    print(f"نسبت عدم تعادل max/min ≈ {imbalance:.1f}×")
    if imbalance > 10:
        print("⚠️ عدم تعادل شدید → از macro-F1 و class_weight='balanced' استفاده کن.")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(counts_train))
    w = 0.4
    b1 = ax.bar(x - w/2, counts_train.values, w, label="train", color=C1)
    b2 = ax.bar(x + w/2, counts_test.values, w, label="test", color=C2)
    ax.bar_label(b1, fontsize=8, padding=1)
    ax.bar_label(b2, fontsize=8, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(counts_train.index.astype(int))
    ax.set_xlabel("Class label")
    ax.set_ylabel("Number of articles")
    ax.set_title("Bar: class distribution (train vs test)")
    ax.legend()
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.95, f"max/min = {imbalance:.1f}×",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=C2, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "01_bar_class_distribution.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 ذخیره شد: {out}")


def plot_length_histogram(train, test):
    section("۲. هیستوگرام — توزیع طول متن (Text Length)")

    for name, df in (("train", train), ("test", test)):
        s = df["n_words"]
        print(f"\n{name}: mean={s.mean():.0f} | median={s.median():.0f} | "
              f"p90={s.quantile(.90):.0f} | p99={s.quantile(.99):.0f} | "
              f"max={s.max()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, (name, df), color in zip(
        axes, [("train", train), ("test", test)], [C1, C2]
    ):
        s = df["n_words"]
        ax.hist(s, bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(s.median(), color=C3, linestyle="--", linewidth=2,
                   label=f"median = {s.median():.0f}")
        ax.axvline(s.mean(), color="black", linestyle=":", linewidth=1.5,
                   label=f"mean = {s.mean():.0f}")
        ax.set_yscale("log")
        ax.set_title(f"Histogram: article length — {name}")
        ax.set_xlabel("Number of words")
        ax.set_ylabel("Frequency (log)")
        ax.legend(fontsize=9)
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIG_DIR / "02_hist_length.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 ذخیره شد: {out}")

    short = (train["n_words"] < MIN_WORDS).sum()
    print(f"\nمتن‌های کوتاه‌تر از {MIN_WORDS} کلمه در train: {short}")


def plot_box_length_by_class(train):
    section("۳. جعبه‌ای — توزیع طول متن به تفکیک کلاس (Box)")

    labels = sorted(train["labels"].unique())
    data = [train.loc[train["labels"] == c, "n_words"].values for c in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # اصلاح نهایی برای سازگاری با همه نسخه‌های Matplotlib
    bp = ax.boxplot(
        data, patch_artist=True,
        boxprops=dict(facecolor="#c7d2fe", color=C1),
        whiskerprops=dict(color=C1),
        capprops=dict(color=C1),
        medianprops=dict(color=C2, linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=C3,
                        markeredgecolor="none", markersize=3, alpha=0.5),
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    
    ax.set_title("Box: article length by class")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Number of words (log)")
    ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIG_DIR / "03_box_length_by_class.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 ذخیره شد: {out}")

    for c in labels:
        m = train.loc[train["labels"] == c, "n_words"].median()
        print(f"  کلاس {c}: median = {m:.0f} کلمه")


def plot_scatter_class_vs_length(train):
    section("۴. پراکندگی — کلاس در برابر طول متن (Scatter)")

    rng = np.random.default_rng(RANDOM_STATE)
    jitter = rng.normal(0, 0.08, len(train))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        train["labels"] + jitter,
        train["n_words"],
        c=train["labels"], cmap="tab10",
        s=8, alpha=0.45, edgecolors="none",
    )
    ax.set_title("Scatter: class vs length (jittered)")
    ax.set_xlabel("Class label (jitter added)")
    ax.set_ylabel("Number of words (log)")
    ax.set_yscale("log")
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIG_DIR / "04_scatter_class_length.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 ذخیره شد: {out}")


FA_STOPWORDS = {
    "بررسی", "مطالعه", "تاثیر", "تأثیر", "تحلیل", "مقاله", "پژوهش",
    "نشان", "دهد", "می", "است", "بود", "شد", "روش", "داده", "ها",
    "های", "را", "که", "این", "آن", "برای", "با", "از", "در", "به",
    "و", "یا", "نیز", "هم", "اما", "اگر", "تا", "بر", "یک", "دو",
    "شده", "شوند", "کرد", "کرده", "باشد", "باشند", "دارد", "دارند",
    "هدف", "پایه", "اساس", "نتایج", "یافته", "حاضر", "مورد", "طور",
}


def plot_top_words(train, top_n: int = 20):
    section(f"۵. نمودار میله‌ای افقی — {top_n} کلمه پرتکرار (سؤال ۳)")

    tokens = re.findall(r"[\w\u0600-\u06FF]+", " ".join(train["sentences"].astype(str)))
    freq = Counter(
        w for w in tokens
        if w not in FA_STOPWORDS and len(w) > 3
    )
    top = freq.most_common(top_n)
    if not top:
        print("⚠️ هیچ کلمه معناداری پیدا نشد.")
        return

    words, counts = zip(*top[::-1])

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(words, counts, color=C1, alpha=0.9)
    for b in bars[-5:]:
        b.set_color(C2)
    ax.set_title(f"Top-{top_n} frequent words (stopwords removed)")
    ax.set_xlabel("Frequency")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fontsize=8, padding=2)

    fig.tight_layout()
    out = FIG_DIR / f"05_top_{top_n}_words.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 ذخیره شد: {out}")

    print(f"\n{top_n} کلمه پرتکرار (پس از حذف stopwords):")
    for w, c in top[:10]:
        print(f"  {w!r:>20}  →  {c}")


def plot_precision_at_k():
    section(f"۶. نمودار میله‌ای — Precision@{TOP_K} هر پرس‌وجو")

    path = Path("evaluation_results.csv")
    if not path.exists():
        print(f"⚠️ «{path}» پیدا نشد؛ این نمودار نیاز به خروجی evaluation.py دارد.")
        print("   ابتدا evaluation.py را اجرا کن.")
        return

    results = pd.read_csv(path)
    scores = (
        results.groupby("query", group_keys=False)
        .apply(lambda g: g.sort_values("rank").head(TOP_K)["relevant"].sum() / TOP_K)
        .sort_values(ascending=False)
    )

    colors = [C1 if s >= scores.mean() else C2 for s in scores.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(scores)), scores.values, color=colors)
    ax.axhline(scores.mean(), color=C3, linestyle="--", linewidth=2,
               label=f"mean = {scores.mean():.2f}")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(scores.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Query")
    ax.set_ylabel(f"Precision@{TOP_K}")
    ax.set_title(f"Bar: Precision@{TOP_K} per query (blue ≥ mean, pink < mean)")
    ax.legend()
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIG_DIR / f"06_precision_at_{TOP_K}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 ذخیره شد: {out}")
    print(f"میانگین Precision@{TOP_K} = {scores.mean():.3f}")


def print_quality_summary(train, test):
    section("۷. خلاصه کیفیت داده (سؤال ۴)")

    def quality_row(name, df):
        s = df["sentences"].astype(str)
        return {
            "مجموعه": name,
            "تعداد ردیف": len(df),
            "گمشده": int(df.isna().sum().sum()),
            "تکراری": int(df.duplicated(subset="sentences").sum()),
            "خالی": int((s.str.strip() == "").sum()),
            f"کوتاه (<{MIN_WORDS}w)": int((df["n_words"] < MIN_WORDS).sum()),
        }

    summary = pd.DataFrame([
        quality_row("train", train),
        quality_row("test", test),
    ])
    print(summary.to_string(index=False))

    overlap = set(train["sentences"].astype(str)) & set(test["sentences"].astype(str))
    print(f"\nمتن‌های مشترک train/test (نشتی): {len(overlap)}")
    if overlap:
        print("⚠️ اگر check_dataset.py را با DROP_LEAKAGE=True اجرا نکرده‌ای، این را اصلاح کن.")

    summary.to_csv(FIG_DIR / "quality_summary.csv", index=False, encoding="utf-8-sig")
    print(f"💾 ذخیره شد: {FIG_DIR/'quality_summary.csv'}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)

    try:
        print("=" * 68)
        print(" EDA — سیستم پیشنهاددهنده مقاله (جلسه ۴، اسلاید ۳۵)")
        print("=" * 68)

        train, test = load_data()

        plot_class_distribution(train, test)
        plot_length_histogram(train, test)
        plot_box_length_by_class(train)
        plot_scatter_class_vs_length(train)
        plot_top_words(train, top_n=20)
        plot_precision_at_k()
        print_quality_summary(train, test)

        section("✅ پایان EDA")
        print(f"📁 همه نمودارها در پوشه «{FIG_DIR}/» ذخیره شدند:")
        for f in sorted(FIG_DIR.glob("*.png")):
            print(f"   • {f.name}  ({f.stat().st_size/1024:.0f} KB)")
    finally:
        sys.stdout = sys.__stdout__
        log.close()
        print(f"\n📄 لاگ کامل در «{LOG_FILE}» ذخیره شد.")


if __name__ == "__main__":
    main()