# %% [markdown]
# # Customer Review Sentiment Analysis — UrbanNest Online Store
#
# **Author:** Jatin Kumar Puppala
#
# ## Business problem
# UrbanNest (a fictional online home & electronics retailer) receives thousands of
# free-text reviews per year. Star ratings tell leadership *that* satisfaction is
# slipping in some places, but not *why*. Nobody has time to read 6,000 reviews.
#
# **Questions from stakeholders**
# 1. How do customers feel overall, and which product categories are under-performing?
# 2. Can we automatically classify review sentiment accurately enough to monitor it weekly?
# 3. **What specifically** are customers unhappy about — product, delivery, price, support,
#    packaging or assembly — and who owns the fix?
# 4. Are any of these problems getting worse over time?
#
# **Approach:** data cleaning → exploratory analysis → lexicon sentiment (VADER) →
# supervised ML classifier (TF-IDF + Logistic Regression) → aspect-based sentiment →
# trend analysis → recommendations.
#
# > Data note: the dataset is synthetic, produced by `src/generate_reviews.py`, and modelled
# > on real e-commerce review patterns. See the README for details.

# %%
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(ROOT / "src"))
from aspects import explode_aspects  # noqa: E402
from domain_vader import DomainVader  # noqa: E402

IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)

# Consistent, colour-blind-friendly styling
NEG, NEU, POS, ACCENT = "#d1495b", "#9aa5b1", "#2e86ab", "#3d5a80"
SENT_COLORS = {"Negative": NEG, "Neutral": NEU, "Positive": POS}
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_colwidth", 140)


def save(fig, name):
    fig.savefig(IMG / f"{name}.png", bbox_inches="tight", dpi=150)

# %% [markdown]
# ## 1. Load and clean the data

# %%
raw = pd.read_csv(ROOT / "data" / "raw" / "reviews.csv", parse_dates=["review_date"])
print(f"Raw shape: {raw.shape}")
raw.head()

# %%
quality = pd.DataFrame({
    "check": ["Duplicate rows", "Empty review text", "Missing region", "Ratings outside 1-5"],
    "rows_affected": [
        raw.duplicated().sum(),
        (raw["review_text"].fillna("").str.strip() == "").sum(),
        raw["region"].isna().sum(),
        (~raw["rating"].between(1, 5)).sum(),
    ],
})
quality

# %%
df = (raw.drop_duplicates()
         .assign(review_text=lambda d: d["review_text"].fillna("").str.strip())
         .query("review_text != ''")
         .assign(region=lambda d: d["region"].fillna("Unknown"))
         .reset_index(drop=True))

# Target label derived from the star rating (industry-standard convention)
df["sentiment"] = pd.cut(df["rating"], bins=[0, 2, 3, 5], labels=["Negative", "Neutral", "Positive"])
df["word_count"] = df["review_text"].str.split().str.len()
df["month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()

print(f"Clean shape: {df.shape}  (removed {len(raw) - len(df)} rows)")

# %% [markdown]
# ## 2. Exploratory analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rating_counts = df["rating"].value_counts().sort_index()
axes[0].bar(rating_counts.index, rating_counts.values,
            color=[NEG, NEG, NEU, POS, POS])
axes[0].set(title="Rating distribution", xlabel="Stars", ylabel="Reviews")
for x, y in zip(rating_counts.index, rating_counts.values):
    axes[0].text(x, y + 40, f"{y / len(df):.0%}", ha="center", fontsize=9)

cat = (df.groupby("category")
         .agg(avg_rating=("rating", "mean"),
              pct_negative=("sentiment", lambda s: (s == "Negative").mean()),
              reviews=("review_id", "count"))
         .sort_values("pct_negative"))
axes[1].barh(cat.index, cat["pct_negative"], color=ACCENT)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[1].set(title="Share of negative (1–2★) reviews by category", xlabel="")
for i, v in enumerate(cat["pct_negative"]):
    axes[1].text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=9)
fig.tight_layout()
save(fig, "01_ratings_overview")
plt.show()
cat.round(3)

# %%
monthly = df.groupby("month").agg(reviews=("review_id", "count"), avg_rating=("rating", "mean"))
fig, ax1 = plt.subplots(figsize=(12, 3.8))
ax1.bar(monthly.index, monthly["reviews"], width=20, color="#d9e2ec", label="Reviews")
ax1.set_ylabel("Reviews per month")
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["avg_rating"], color=ACCENT, marker="o", lw=2, label="Avg rating")
ax2.set_ylabel("Average rating")
ax2.grid(False)
ax1.set_title("Review volume and average rating by month")
fig.tight_layout()
save(fig, "02_monthly_volume_rating")
plt.show()

# %% [markdown]
# ## 3. Lexicon-based sentiment with VADER
# VADER is a rule-based sentiment model designed for short, informal text. It needs no
# training data, which makes it a quick baseline. It returns a **compound score** from
# −1 (very negative) to +1 (very positive).

#
# ### Adapting VADER to retail language
# Stock VADER was built for social media. Reading a sample of reviews showed it misses
# many retail complaints:

# %%
stock_vader, vader = DomainVader(use_domain=False), DomainVader(use_domain=True)
examples = [
    "Customer service never responded to my emails.",
    "Way overpriced for what you get.",
    "The bookshelf stopped working after two weeks.",
    "Cheap materials, the lamp feels flimsy and poorly made.",
    "Contacted support three times and got three different answers.",
]
pd.DataFrame({"sentence": examples,
              "stock_VADER": [stock_vader.compound(s) for s in examples],
              "domain_VADER": [vader.compound(s) for s in examples]}).round(2)

# %% [markdown]
# `src/domain_vader.py` adds about 40 retail-specific words and phrases to VADER's lexicon
# ("overpriced", "flimsy", "stopped working", "never responded"). It also neutralises
# "support", which stock VADER treats as positive even when it is just the name of a
# department. All further VADER scores use this domain version.

# %%
to_label = lambda s: pd.cut(s, bins=[-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"])
df["stock_vader_label"] = to_label(df["review_text"].apply(stock_vader.compound))
df["vader_compound"] = df["review_text"].apply(vader.compound)
df["vader_label"] = to_label(df["vader_compound"])

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="rating", y="vader_compound", color="#d9e2ec", ax=ax, fliersize=1.5)
ax.axhline(0, color="grey", lw=1, ls="--")
ax.set(title="VADER compound score vs. star rating", xlabel="Stars", ylabel="VADER compound")
fig.tight_layout()
save(fig, "03_vader_vs_rating")
plt.show()

print(f"Spearman correlation (VADER vs stars): "
      f"{df[['vader_compound', 'rating']].corr(method='spearman').iloc[0, 1]:.2f}")
print(f"VADER accuracy vs rating-based label: {accuracy_score(df['sentiment'], df['vader_label']):.1%}")

# %% [markdown]
# ## 4. Supervised sentiment classifier
# The VADER baseline is fast but generic. A model **trained on our own reviews** learns
# domain language such as "holes don't line up" or "marked as delivered". We compare:
#
# * **VADER**: rule-based baseline
# * **Multinomial Naive Bayes**: classic text baseline
# * **Logistic Regression**: TF-IDF unigrams + bigrams with class weighting
#
# Neutral (3★) is the minority class, so **macro-F1** is the headline metric. It weights
# every class equally rather than rewarding a model that only predicts "Positive".

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"], df["sentiment"].astype(str), test_size=0.2,
    stratify=df["sentiment"], random_state=42)

# Remove common stop words but KEEP negations: "not recommend" must not collapse to "recommend"
NEGATIONS = {"not", "no", "nor", "never", "nothing", "cannot", "against"}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - NEGATIONS)

tfidf_args = dict(lowercase=True, ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                  stop_words=STOP_WORDS, token_pattern=r"(?u)\b[a-zA-Z']{2,}\b")
models = {
    "Naive Bayes": make_pipeline(TfidfVectorizer(**tfidf_args), MultinomialNB(alpha=0.3)),
    "Logistic Regression": make_pipeline(
        TfidfVectorizer(**tfidf_args),
        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
}

results = []
for name, col in [("VADER (stock)", "stock_vader_label"), ("VADER (domain-tuned)", "vader_label")]:
    pred = df.loc[X_test.index, col].astype(str)
    results.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro")})
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro")})

results = pd.DataFrame(results).set_index("model").sort_values("macro_f1")
results.style.format("{:.1%}")

# %%
best = models["Logistic Regression"]
y_pred = best.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
sns.heatmap(cm, annot=True, fmt=".0%", cmap="Blues", cbar=False, xticklabels=labels,
            yticklabels=labels, ax=axes[0])
axes[0].set(title="Logistic Regression — confusion matrix\n(row-normalised)",
            xlabel="Predicted", ylabel="Actual")

results.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"}).plot.barh(
    ax=axes[1], color=[NEU, ACCENT], width=0.75)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[1].set(title="Model comparison (test set)", xlabel="", ylabel="", xlim=(0, 1))
for cont in axes[1].containers:
    axes[1].bar_label(cont, fmt=lambda v: f"{v:.1%}", padding=3, fontsize=8.5)
axes[1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, "04_model_performance")
plt.show()

# %% [markdown]
# ### What drives the model? The most influential words and phrases
# Logistic Regression is interpretable: each word or phrase has a coefficient per class.
# The strongest negative-class terms read like a **list of operational failures**.

# %%
vec, clf = best.named_steps["tfidfvectorizer"], best.named_steps["logisticregression"]
terms = np.array(vec.get_feature_names_out())
coef = pd.DataFrame(clf.coef_.T, index=terms, columns=clf.classes_)

top_n = 12
top_neg = coef["Negative"].nlargest(top_n)[::-1]
top_pos = coef["Positive"].nlargest(top_n)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].barh(top_neg.index, top_neg.values, color=NEG)
axes[0].set_title("Strongest NEGATIVE signals")
axes[1].barh(top_pos.index, top_pos.values, color=POS)
axes[1].set_title("Strongest POSITIVE signals")
for ax in axes:
    ax.set_xlabel("Model coefficient")
fig.tight_layout()
save(fig, "05_top_terms")
plt.show()

# %% [markdown]
# ### Error analysis
# Where does the model still get it wrong? Reviewing misclassifications is how a model
# gets improved in practice.

# %%
errors = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})
errors = errors[errors.actual != errors.predicted]
print(f"{len(errors)} misclassified out of {len(y_test)} test reviews")
errors.sample(6, random_state=7)

# %% [markdown]
# **Takeaway:** most errors involve the **Neutral** class. These are 3★ reviews that
# mix praise and complaints (for example, great quality but slow delivery). The text
# carries mixed signals and the boundary between 3★ and 4★ is subjective even for
# people. Scoring each aspect separately (next section) handles this better than one
# label per review.

# %% [markdown]
# ## 5. Aspect-based sentiment: *what* are customers unhappy about?
# Each review is split into sentences, and each sentence is tagged with the business area
# it mentions (rule-based keyword dictionary in `src/aspects.py`) and scored with VADER.
# This turns unstructured text into a table operations teams can act on.

# %%
asp = explode_aspects(df, id_cols=("review_id", "category", "month", "review_date"))
asp["compound"] = asp["sentence"].apply(vader.compound)
asp["is_negative"] = asp["compound"] <= -0.05
print(f"{len(asp):,} aspect mentions extracted from {asp.review_id.nunique():,} reviews "
      f"({asp.review_id.nunique() / len(df):.0%} coverage)")
asp.sample(5, random_state=3)[["sentence", "aspect", "compound"]]

# %% [markdown]
# Why domain tuning matters here: with stock VADER, complaints about **customer service**
# and **price** are badly under-counted. That would have sent leadership the wrong message.

# %%
asp["stock_negative"] = asp["sentence"].apply(stock_vader.compound) <= -0.05
(asp.groupby("aspect")[["stock_negative", "is_negative"]].mean()
    .rename(columns={"stock_negative": "% negative (stock VADER)", "is_negative": "% negative (domain VADER)"})
    .sort_values("% negative (domain VADER)", ascending=False)
    .style.format("{:.1%}"))

# %%
summary = (asp.groupby("aspect")
              .agg(mentions=("review_id", "nunique"),
                   pct_negative=("is_negative", "mean"),
                   avg_sentiment=("compound", "mean"))
              .assign(share_of_reviews=lambda d: d["mentions"] / len(df))
              .sort_values("pct_negative", ascending=False))
# "Negative mentions" = how many reviews complain about this aspect: the priority metric
summary["negative_mentions"] = (asp[asp.is_negative].groupby("aspect")["review_id"].nunique()
                                .reindex(summary.index))

fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.scatter(summary["share_of_reviews"], summary["pct_negative"],
           s=summary["negative_mentions"] * 1.4, color=ACCENT, alpha=0.55, edgecolor="white")
for name, r in summary.iterrows():
    ax.annotate(f"{name}\n{int(r.negative_mentions):,} complaints",
                (r.share_of_reviews, r.pct_negative), ha="center", va="center", fontsize=8.5)
ax.axhline(summary["pct_negative"].median(), color="grey", ls="--", lw=1)
ax.axvline(summary["share_of_reviews"].median(), color="grey", ls="--", lw=1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set(title="Customer pain-point matrix\n(bubble size = number of complaining reviews)",
       xlabel="How often it's mentioned (share of reviews)",
       ylabel="How often it's negative when mentioned")
ax.margins(0.2)
fig.tight_layout()
save(fig, "06_pain_point_matrix")
plt.show()
summary.round(3)

# %%
heat = (asp.pivot_table(index="category", columns="aspect", values="is_negative", aggfunc="mean"))
counts = asp.pivot_table(index="category", columns="aspect", values="is_negative", aggfunc="size")
heat = heat.where(counts >= 30)   # hide cells with too few mentions to be reliable

fig, ax = plt.subplots(figsize=(11, 4.2))
sns.heatmap(heat, annot=True, fmt=".0%", cmap="Reds", vmin=0, vmax=0.6,
            linewidths=0.5, cbar_kws={"label": "% of mentions that are negative"}, ax=ax)
ax.set(title="Where are the problems? Negative share by category × aspect",
       xlabel="", ylabel="")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
fig.tight_layout()
save(fig, "07_category_aspect_heatmap")
plt.show()

# %% [markdown]
# ## 6. Trends: are problems getting better or worse?

# %%
def monthly_negative_share(frame, aspect, category=None):
    f = frame[frame.aspect == aspect]
    if category:
        f = f[f.category == category]
    return f.groupby("month")["is_negative"].mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
series = [
    ("Delivery & Shipping", None, "Delivery complaints (all categories)"),
    ("Assembly & Ease of Use", "Furniture", "Furniture assembly complaints"),
    ("Packaging", None, "Packaging complaints (all categories)"),
]
for ax, (aspect, cat_, title) in zip(axes, series):
    s = monthly_negative_share(asp, aspect, cat_)
    ax.plot(s.index, s.values, color=ACCENT, lw=1.2, alpha=0.5)
    ax.plot(s.index, s.rolling(3, min_periods=1).mean(), color=NEG, lw=2.4, label="3-mo rolling avg")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis="x", rotation=45)
axes[0].set_ylabel("% of mentions negative")
axes[0].legend(loc="upper left", fontsize=8)
# Highlight the holiday peaks
for yr in (2024, 2025):
    axes[0].axvspan(pd.Timestamp(f"{yr}-11-01"), pd.Timestamp(f"{yr}-12-31"), color=NEG, alpha=0.08)
axes[1].axvline(pd.Timestamp("2025-04-01"), color="grey", ls="--", lw=1)
axes[1].text(pd.Timestamp("2025-04-10"), 0.05, "Q2 2025", fontsize=8, color="grey")
axes[2].axvline(pd.Timestamp("2024-07-01"), color="grey", ls="--", lw=1)
axes[2].text(pd.Timestamp("2024-07-10"), 0.05, "Jul 2024", fontsize=8, color="grey")
fig.tight_layout()
save(fig, "08_aspect_trends")
plt.show()

# %%
# Quantify the changes behind the charts
d = asp[asp.aspect == "Delivery & Shipping"].assign(
    holiday=lambda f: f.month.dt.month.isin([11, 12]), year=lambda f: f.month.dt.year)
holiday_tbl = d.pivot_table(index="year", columns="holiday", values="is_negative", aggfunc="mean")
holiday_tbl.columns = ["Jan–Oct", "Nov–Dec (holiday)"]
print("Delivery: % negative mentions")
display(holiday_tbl.style.format("{:.1%}"))

f = asp[(asp.aspect == "Assembly & Ease of Use") & (asp.category == "Furniture")]
furn = f.groupby(f.review_date.dt.to_period("Q"))["is_negative"].mean()
print("Furniture assembly: % negative mentions by quarter")
display(furn.to_frame("pct_negative").T.style.format("{:.0%}"))

p = asp[asp.aspect == "Packaging"]
before = p[p.review_date < "2024-07-01"].is_negative.mean()
after = p[p.review_date >= "2024-07-01"].is_negative.mean()
print(f"Packaging: {before:.1%} negative before Jul 2024 -> {after:.1%} after")

# %% [markdown]
# ## 7. Export scored data
# The scored reviews can feed a BI dashboard (Power BI or Tableau) for weekly monitoring.

# %%
out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
df.assign(predicted_sentiment=best.predict(df["review_text"])).to_csv(out / "reviews_scored.csv", index=False)
asp.to_csv(out / "aspect_mentions.csv", index=False)
print("Saved processed files to", out.relative_to(ROOT))

# %% [markdown]
# ## 8. Key insights and recommendations
#
# | # | Finding | Evidence | Recommendation | Owner |
# |---|---------|----------|----------------|-------|
# | 1 | **Holiday delivery failures are getting worse.** | Negative delivery mentions jump from ~27% (Jan–Oct) to **49% in Nov–Dec 2024** and **65% in Nov–Dec 2025**. | Add peak-season carrier capacity from October, set more conservative delivery estimates in Nov–Dec, and send proactive delay notifications. | Logistics |
# | 2 | **Furniture assembly problems roughly doubled from Q2 2025.** | Negative assembly mentions rose from **37% to 65%**. Complaints cite misaligned holes, missing screws and unclear instructions. The timing points to a supplier or design change. | Audit Q2 2025 furniture suppliers, add a hardware checklist at packing, and publish video assembly guides. | Merchandising / Supplier QA |
# | 3 | **Electronics has a customer-support problem.** | **58%** of Electronics service mentions are negative, against 35–43% in other categories. | Set up a dedicated Electronics support queue with a 24-hour response SLA, and track first-contact resolution. | Customer Service |
# | 4 | **Product quality is the largest driver of negative reviews.** | **70%** of 1–2★ reviews include a negative quality sentence. Delivery follows at 31%. | Flag products whose share of negative quality mentions exceeds the category baseline for vendor review each month. | Category Management |
# | 5 | **The July 2024 packaging redesign worked.** | Negative packaging mentions fell from **27% to 12%** after the change. | Treat it as a success case and use the same before/after measurement for future operational changes. | Operations |
# | 6 | **Price is not the problem.** | Only **11%** of price mentions are negative, the lowest of any aspect. | Avoid discounting in response to falling ratings. Fix fulfilment and quality first. | Pricing / Marketing |
#
# ### Model takeaways
# * **TF-IDF + Logistic Regression** is the chosen model, with the best macro-F1 (~0.71)
#   at ~78% accuracy. Naive Bayes has slightly *higher* accuracy but leans towards the
#   majority Positive class and misses more Negative and Neutral reviews. Those are the
#   reviews the business most needs to catch, which is why accuracy alone is the wrong
#   metric here.
# * Neutral (3★) reviews are hardest to classify because they mix praise and complaints.
# * **Off-the-shelf tools can mislead.** Stock VADER flagged fewer than 1 in 10 customer
#   service mentions as negative. After domain tuning, the figure is about **46%**. Without
#   that step, the service problem in finding 3 would have stayed hidden.
#
# ### Next steps
# * Deploy the classifier as a weekly batch job feeding a BI dashboard, with alerts when an
#   aspect's negative share moves more than 10 percentage points above its 3-month average.
# * Replace the keyword-based aspect tagger with a zero-shot or fine-tuned transformer model
#   (for example, a BERT-family model) once labelled sentence-level data is available.
