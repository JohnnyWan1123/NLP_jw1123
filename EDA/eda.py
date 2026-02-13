"""
Stage 2 - EDA on the Don't Patronize Me PCL dataset.
Covers class balance / text length stats and a lexical comparison between PCL and non-PCL.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Load data

df = pd.read_csv(
    "dontpatronizeme_pcl.tsv",
    sep="\t",
    skiprows=4,
    header=None,
    names=["par_id", "art_id", "keyword", "country_code", "text", "label"],
)

# binarise: 0,1 -> not PCL; 2,3,4 -> PCL
df["binary_label"] = (df["label"] >= 2).astype(int)

# quick length features
df["token_count"] = df["text"].apply(lambda x: len(word_tokenize(str(x))))
df["char_count"] = df["text"].apply(lambda x: len(str(x)))
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

print(f"Total samples: {len(df)}")
print(f"PCL (label >= 2): {df['binary_label'].sum()}")
print(f"Not PCL (label < 2): {(df['binary_label'] == 0).sum()}")
print(f"Imbalance ratio: 1:{(df['binary_label'] == 0).sum() // df['binary_label'].sum()}")
print()

# ---- EDA 1: class distribution + text length stats ----

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "EDA Technique 1: Class Distribution & Text Length Profiling",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# binary class bar chart
ax1 = fig.add_subplot(gs[0, 0])
class_counts = df["binary_label"].value_counts().sort_index()
bars = ax1.bar(
    ["Not PCL (0)", "PCL (1)"],
    class_counts.values,
    color=["#4C72B0", "#DD8452"],
    edgecolor="black",
    width=0.5,
)
for bar, count in zip(bars, class_counts.values):
    pct = count / len(df) * 100
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{count}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
ax1.set_title("Binary Class Distribution", fontsize=13, fontweight="bold")
ax1.set_ylabel("Count")
ax1.set_ylim(0, class_counts.max() * 1.2)

# original 0-4 labels
ax2 = fig.add_subplot(gs[0, 1])
label_counts = df["label"].value_counts().sort_index()
colors = ["#4C72B0", "#6BA3D6", "#DD8452", "#E8A838", "#C44E52"]
bars2 = ax2.bar(
    label_counts.index.astype(str),
    label_counts.values,
    color=colors[: len(label_counts)],
    edgecolor="black",
    width=0.5,
)
for bar, count in zip(bars2, label_counts.values):
    pct = count / len(df) * 100
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{count}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax2.set_title("Fine-Grained Label Distribution (0–4)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Label")
ax2.set_ylabel("Count")
ax2.set_ylim(0, label_counts.max() * 1.2)

# token length histograms per class
ax3 = fig.add_subplot(gs[1, 0])
pcl_tokens = df[df["binary_label"] == 1]["token_count"]
no_pcl_tokens = df[df["binary_label"] == 0]["token_count"]
ax3.hist(
    no_pcl_tokens,
    bins=60,
    alpha=0.6,
    label=f"Not PCL (μ={no_pcl_tokens.mean():.1f})",
    color="#4C72B0",
    edgecolor="black",
    density=True,
)
ax3.hist(
    pcl_tokens,
    bins=60,
    alpha=0.6,
    label=f"PCL (μ={pcl_tokens.mean():.1f})",
    color="#DD8452",
    edgecolor="black",
    density=True,
)
ax3.axvline(no_pcl_tokens.mean(), color="#4C72B0", linestyle="--", linewidth=2)
ax3.axvline(pcl_tokens.mean(), color="#DD8452", linestyle="--", linewidth=2)
ax3.set_title("Token Count Distribution by Class", fontsize=13, fontweight="bold")
ax3.set_xlabel("Token Count")
ax3.set_ylabel("Density")
ax3.legend(fontsize=10)
ax3.set_xlim(0, df["token_count"].quantile(0.99))

# stats table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
stats_data = []
for label_name, subset in [("Not PCL", df[df["binary_label"] == 0]),
                            ("PCL", df[df["binary_label"] == 1]),
                            ("All", df)]:
    tc = subset["token_count"]
    stats_data.append([
        label_name,
        f"{len(subset)}",
        f"{tc.mean():.1f}",
        f"{tc.median():.0f}",
        f"{tc.std():.1f}",
        f"{tc.min()}",
        f"{tc.max()}",
        f"{tc.quantile(0.95):.0f}",
    ])
table = ax4.table(
    cellText=stats_data,
    colLabels=["Class", "N", "Mean", "Median", "Std", "Min", "Max", "P95"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F2F2F2")
ax4.set_title("Token Count Summary Statistics", fontsize=13, fontweight="bold", pad=20)

plt.savefig("eda_technique1_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# ---- EDA 2: lexical comparison (unigrams, bigrams, distinctive words) ----

stop_words = set(stopwords.words("english"))


def clean_tokenize(text):
    """Lowercase, strip punctuation, drop stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def get_top_ngrams(texts, n=2, top_k=20):
    """Return the top_k most common n-grams."""
    all_ngrams = []
    for text in texts:
        tokens = clean_tokenize(text)
        all_ngrams.extend(list(ngrams(tokens, n)))
    return Counter(all_ngrams).most_common(top_k)


pcl_texts = df[df["binary_label"] == 1]["text"].tolist()
no_pcl_texts = df[df["binary_label"] == 0]["text"].tolist()

# Unigrams
pcl_unigrams = Counter()
no_pcl_unigrams = Counter()
for text in pcl_texts:
    pcl_unigrams.update(clean_tokenize(text))
for text in no_pcl_texts:
    no_pcl_unigrams.update(clean_tokenize(text))

# Bigrams
pcl_bigrams = get_top_ngrams(pcl_texts, n=2, top_k=20)
no_pcl_bigrams = get_top_ngrams(no_pcl_texts, n=2, top_k=20)

# words that appear disproportionately often in PCL (ratio of relative freqs)
total_pcl = sum(pcl_unigrams.values())
total_no_pcl = sum(no_pcl_unigrams.values())
distinctive_pcl = {}
for word, count in pcl_unigrams.items():
    if count >= 10:  # minimum frequency threshold
        freq_pcl = count / total_pcl
        freq_no_pcl = (no_pcl_unigrams.get(word, 0) + 1) / total_no_pcl  # smoothing
        distinctive_pcl[word] = freq_pcl / freq_no_pcl

top_distinctive = sorted(distinctive_pcl.items(), key=lambda x: x[1], reverse=True)[:20]

# plots
fig2 = plt.figure(figsize=(20, 16))
fig2.suptitle(
    "EDA Technique 2: Comparative Lexical Analysis (PCL vs Not-PCL)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)
gs2 = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# top unigrams side by side
ax_a = fig2.add_subplot(gs2[0, 0])
top_pcl_words = pcl_unigrams.most_common(15)
words_pcl, counts_pcl = zip(*top_pcl_words)
y_pos = range(len(words_pcl))
ax_a.barh(y_pos, counts_pcl, color="#DD8452", edgecolor="black")
ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(words_pcl, fontsize=10)
ax_a.invert_yaxis()
ax_a.set_title("Top 15 Unigrams in PCL Texts", fontsize=13, fontweight="bold")
ax_a.set_xlabel("Frequency")

ax_b = fig2.add_subplot(gs2[0, 1])
top_nopcl_words = no_pcl_unigrams.most_common(15)
words_nopcl, counts_nopcl = zip(*top_nopcl_words)
y_pos2 = range(len(words_nopcl))
ax_b.barh(y_pos2, counts_nopcl, color="#4C72B0", edgecolor="black")
ax_b.set_yticks(y_pos2)
ax_b.set_yticklabels(words_nopcl, fontsize=10)
ax_b.invert_yaxis()
ax_b.set_title("Top 15 Unigrams in Not-PCL Texts", fontsize=13, fontweight="bold")
ax_b.set_xlabel("Frequency")

# words most overrepresented in PCL
ax_c = fig2.add_subplot(gs2[1, 0])
dist_words, dist_ratios = zip(*top_distinctive)
y_pos3 = range(len(dist_words))
ax_c.barh(y_pos3, dist_ratios, color="#C44E52", edgecolor="black")
ax_c.set_yticks(y_pos3)
ax_c.set_yticklabels(dist_words, fontsize=10)
ax_c.invert_yaxis()
ax_c.set_title(
    "Most Distinctive PCL Words\n(Relative Frequency Ratio: PCL / Not-PCL)",
    fontsize=13,
    fontweight="bold",
)
ax_c.set_xlabel("Frequency Ratio")

# bigram table
ax_d = fig2.add_subplot(gs2[1, 1])
ax_d.axis("off")
bigram_data = []
for i in range(min(15, len(pcl_bigrams), len(no_pcl_bigrams))):
    pcl_bg = " ".join(pcl_bigrams[i][0])
    pcl_ct = pcl_bigrams[i][1]
    nopcl_bg = " ".join(no_pcl_bigrams[i][0])
    nopcl_ct = no_pcl_bigrams[i][1]
    bigram_data.append([f"{pcl_bg} ({pcl_ct})", f"{nopcl_bg} ({nopcl_ct})"])

table2 = ax_d.table(
    cellText=bigram_data,
    colLabels=["PCL Bigrams (count)", "Not-PCL Bigrams (count)"],
    loc="center",
    cellLoc="center",
)
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1.0, 1.6)
for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor("#FDEBD0")
    else:
        cell.set_facecolor("#D6EAF8")
ax_d.set_title("Top 15 Bigrams: PCL vs Not-PCL", fontsize=13, fontweight="bold", pad=20)

plt.savefig("eda_technique2_lexical_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# how does PCL rate differ across the keyword/community groups?
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle(
    "EDA Technique 2 (cont.): Keyword/Community Distribution by Class",
    fontsize=14,
    fontweight="bold",
)

keyword_class = df.groupby(["keyword", "binary_label"]).size().unstack(fill_value=0)
keyword_class.columns = ["Not PCL", "PCL"]
keyword_class["pcl_rate"] = keyword_class["PCL"] / (keyword_class["PCL"] + keyword_class["Not PCL"])
keyword_class = keyword_class.sort_values("pcl_rate", ascending=True)

# Stacked bar
keyword_class[["Not PCL", "PCL"]].plot(
    kind="barh", stacked=True, ax=axes3[0],
    color=["#4C72B0", "#DD8452"], edgecolor="black"
)
axes3[0].set_title("Samples per Keyword & Class", fontsize=12, fontweight="bold")
axes3[0].set_xlabel("Count")

# PCL rate
keyword_class["pcl_rate"].plot(
    kind="barh", ax=axes3[1], color="#C44E52", edgecolor="black"
)
axes3[1].set_title("PCL Rate by Keyword/Community", fontsize=12, fontweight="bold")
axes3[1].set_xlabel("PCL Rate")
axes3[1].axvline(df["binary_label"].mean(), color="black", linestyle="--", label="Overall PCL rate")
axes3[1].legend()

plt.tight_layout()
plt.savefig("eda_technique2_keyword_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
