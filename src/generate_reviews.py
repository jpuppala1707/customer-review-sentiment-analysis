"""
Generate a synthetic customer-review dataset for "UrbanNest", a fictional
online home & electronics retailer.

Why synthetic? Real review datasets are either licensed or heavily cleaned.
Generating the data lets me plant realistic business problems (a holiday
delivery crunch, a furniture assembly issue, an electronics support backlog)
and then test whether the NLP pipeline can *find* them without being told.

Each review is built from a set of latent "aspect" sentiments
(quality, delivery, price, service, packaging, assembly / ease of use).
The star rating is derived from those latent sentiments plus noise, so the
text and the rating are related but not perfectly aligned - like real life.

Usage:
    python src/generate_reviews.py            # writes data/raw/reviews.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_REVIEWS = 6000
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "reviews.csv"

CATEGORIES = {
    "Kitchen Appliances": ["air fryer", "blender", "coffee maker", "toaster", "stand mixer", "kettle"],
    "Electronics": ["bluetooth speaker", "smart plug", "wireless earbuds", "security camera", "tablet stand", "smart bulb"],
    "Furniture": ["bookshelf", "office chair", "TV console", "bed frame", "dining table", "desk"],
    "Home Decor": ["wall mirror", "table lamp", "area rug", "throw pillow set", "picture frame set", "curtains"],
    "Bedding & Bath": ["duvet cover", "bath towel set", "mattress topper", "pillow pair", "sheet set", "bath mat"],
}
CATEGORY_WEIGHTS = [0.24, 0.22, 0.18, 0.18, 0.18]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

# --------------------------------------------------------------------------
# Phrase banks: {aspect: {"pos": [...], "neg": [...], "neu": [...]}}
# "{p}" is replaced with the product noun.
# --------------------------------------------------------------------------
PHRASES = {
    "quality": {
        "pos": [
            "The {p} feels really solid and well made.",
            "Excellent build quality, the {p} looks even better in person.",
            "Great quality for the money, the materials feel premium.",
            "This {p} works perfectly and has held up well after several weeks.",
            "Very sturdy and durable, I'm impressed.",
            "Love it! The {p} is exactly as described.",
            "Works like a charm, no issues at all.",
        ],
        "neg": [
            "The {p} stopped working after two weeks.",
            "Cheap materials, the {p} feels flimsy and poorly made.",
            "Mine arrived with a crack and a loose part.",
            "Quality is terrible, it broke on the first use.",
            "The {p} looks nothing like the photos, very disappointing.",
            "Defective unit, it makes a weird noise and overheats.",
            "Started falling apart within a month.",
        ],
        "neu": [
            "The {p} is okay, does what it says.",
            "Quality is average, nothing special.",
            "It's a standard {p}, about what you'd expect.",
        ],
    },
    "delivery": {
        "pos": [
            "Shipping was fast, it arrived two days early.",
            "Quick delivery and the driver was careful.",
            "Arrived on time and tracking was accurate.",
            "Super fast shipping, ordered Monday and had it Wednesday.",
        ],
        "neg": [
            "Delivery took almost three weeks and tracking never updated.",
            "Shipping was extremely slow, it arrived way past the estimated date.",
            "The package was delayed twice and nobody told me why.",
            "Late delivery, I had to cancel my plans waiting for it.",
            "It was marked as delivered but showed up four days later.",
            "Order was lost in transit and had to be reshipped.",
        ],
        "neu": [
            "Delivery was within the estimated window.",
            "Shipping took about a week.",
        ],
    },
    "price": {
        "pos": [
            "Great value for the price.",
            "Cheaper than other stores and the quality is just as good.",
            "Got it on sale, an absolute bargain.",
            "Worth every penny.",
        ],
        "neg": [
            "Way overpriced for what you get.",
            "Not worth the money at all.",
            "Too expensive compared to similar products.",
            "The price went up right after I bought it, feels like a rip off.",
        ],
        "neu": [
            "Price is fair, roughly the same as elsewhere.",
            "Reasonably priced, not the cheapest though.",
        ],
    },
    "service": {
        "pos": [
            "Customer service was helpful and replaced it right away.",
            "Support answered my question within an hour, very friendly.",
            "The return process was easy and the refund was quick.",
            "Shoutout to the support team, they were fantastic.",
        ],
        "neg": [
            "Customer service never responded to my emails.",
            "I was on hold for over an hour and then the call dropped.",
            "Support was rude and refused to help with the warranty.",
            "Still waiting on my refund after a month, terrible customer service.",
            "Contacted support three times and got three different answers.",
        ],
        "neu": [
            "I contacted support once and they answered eventually.",
        ],
    },
    "packaging": {
        "pos": [
            "Packaging was neat and eco friendly.",
            "Very well packed, everything was protected.",
        ],
        "neg": [
            "The box was crushed and the packaging was torn.",
            "Poor packaging, items were rattling around loose.",
            "Way too much plastic packaging and the box was damaged.",
        ],
        "neu": [
            "Came in a plain brown box.",
        ],
    },
    "usability": {
        "pos": [
            "Setup was easy and the instructions were clear.",
            "Took ten minutes to assemble, very straightforward.",
            "Easy to use, even my parents figured it out.",
            "The app connects instantly and is simple to use.",
        ],
        "neg": [
            "Assembly was a nightmare, the instructions are confusing and holes don't line up.",
            "Missing screws, could not finish the assembly.",
            "The instructions are useless and setup took hours.",
            "The app keeps disconnecting and is hard to use.",
            "Hard to assemble, two of the panels were drilled wrong.",
        ],
        "neu": [
            "Setup took about half an hour.",
            "Assembly is standard, you'll need your own screwdriver.",
        ],
    },
}

# Tricky phrasings that trip up lexicon-based models (negation, sarcasm, contrast)
TRICKY = {
    "pos": [
        "Not bad at all, honestly.",
        "I was skeptical but it's not disappointing in the slightest.",
        "Can't complain.",
    ],
    "neg": [
        "Oh great, another product that breaks in a week.",
        "Wanted to love it, but I just can't.",
        "Thanks for nothing.",
        "Would not buy again.",
    ],
}

OPENERS = ["", "", "", "Bought this for my new apartment. ", "Second time ordering from here. ",
           "Ordered as a gift. ", "Long review but worth reading. ", "Update after a month: "]
CLOSERS = {
    "pos": ["Highly recommend!", "Would buy again.", "5 stars from me.", "Very happy with this purchase.", ""],
    "neg": ["Returning it.", "Avoid.", "Do not recommend.", "Very disappointed.", ""],
    "neu": ["It's fine.", "Might buy again if on sale.", "", ""],
}

# Base probability that a review mentions an aspect, by category
MENTION_P = {
    "Kitchen Appliances": dict(quality=.85, delivery=.35, price=.35, service=.12, packaging=.12, usability=.30),
    "Electronics":        dict(quality=.80, delivery=.30, price=.35, service=.22, packaging=.10, usability=.45),
    "Furniture":          dict(quality=.75, delivery=.45, price=.30, service=.15, packaging=.20, usability=.60),
    "Home Decor":         dict(quality=.85, delivery=.35, price=.40, service=.08, packaging=.22, usability=.08),
    "Bedding & Bath":     dict(quality=.85, delivery=.30, price=.40, service=.08, packaging=.08, usability=.05),
}

# Probability that a mentioned aspect is negative (baseline)
NEG_P = dict(quality=.22, delivery=.20, price=.20, service=.40, packaging=.25, usability=.25)


def neg_probability(aspect: str, category: str, date: pd.Timestamp) -> float:
    """Plant business problems the analysis should uncover."""
    p = NEG_P[aspect]
    # 1) Holiday delivery crunch: Nov-Dec each year, worse in 2025 (new carrier)
    if aspect == "delivery" and date.month in (11, 12):
        p = 0.45 if date.year == 2024 else 0.62
    # 2) Furniture assembly problem (supplier change from Q2 2025)
    if aspect == "usability" and category == "Furniture":
        p = 0.38 if date < pd.Timestamp("2025-04-01") else 0.60
    # 3) Electronics support backlog
    if aspect == "service" and category == "Electronics":
        p = 0.62
    # 4) Packaging redesign in Jul 2024 reduced damage complaints
    if aspect == "packaging" and date >= pd.Timestamp("2024-07-01"):
        p = 0.14
    # 5) Home Decor is the strongest category on quality
    if aspect == "quality" and category == "Home Decor":
        p = 0.12
    return p


def mention_probability(aspect: str, category: str, date: pd.Timestamp) -> float:
    p = MENTION_P[category][aspect]
    if aspect == "delivery" and date.month in (11, 12):
        p = min(0.9, p + 0.25)  # people talk about shipping more during the holidays
    return p


def build_review(rng: random.Random, nrng: np.random.Generator, category: str, product: str,
                 date: pd.Timestamp) -> tuple[str, int, dict]:
    aspects = {}
    for aspect in PHRASES:
        if rng.random() < mention_probability(aspect, category, date):
            r = rng.random()
            pn = neg_probability(aspect, category, date)
            pneu = 0.15
            aspects[aspect] = "neg" if r < pn else ("neu" if r < pn + pneu else "pos")
    if not aspects:  # every review talks about something
        aspects["quality"] = rng.choice(["pos", "pos", "neu", "neg"])

    # Latent satisfaction score -> star rating
    weights = dict(quality=1.6, delivery=0.9, price=0.8, service=1.2, packaging=0.4, usability=1.0)
    val = {"pos": 1.0, "neu": 0.0, "neg": -1.25}
    text_score = sum(weights[a] * val[s] for a, s in aspects.items()) / max(1.0, len(aspects) ** 0.5)
    # The overall tone of the *text* (drives closing remarks) ...
    overall = "pos" if text_score > 0.4 else ("neg" if text_score < -0.4 else "neu")
    # ... while the star rating also reflects personal taste / expectations not in the text
    score = text_score + nrng.normal(0, 0.5)
    rating = int(np.clip(np.round(3.4 + 1.25 * score), 1, 5))
    if rng.random() < 0.03:        # wrong star clicked / rating inconsistent with text
        rating = rng.randint(1, 5)

    sentences = [rng.choice(PHRASES[a][s]).format(p=product) for a, s in aspects.items()]
    rng.shuffle(sentences)
    if overall in TRICKY and rng.random() < 0.12:
        sentences.append(rng.choice(TRICKY[overall]))
    text = rng.choice(OPENERS) + " ".join(sentences)
    closer = rng.choice(CLOSERS[overall])
    if closer:
        text += " " + closer
    # light, realistic noise: lowercase, missing final period, exclamation marks
    if rng.random() < 0.08:
        text = text.lower()
    if rng.random() < 0.05:
        text = text.rstrip(".")
    return text, rating, aspects


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)

    # Seasonal order volume: more reviews in Q4, gentle growth over time
    days = pd.date_range(START, END, freq="D")
    season = 1 + 0.45 * days.month.isin([11, 12]) + 0.15 * days.month.isin([1])
    growth = np.linspace(1.0, 1.35, len(days))
    p_day = (season * growth) / (season * growth).sum()
    review_dates = nrng.choice(days, size=N_REVIEWS, p=p_day)

    # 120 products with stable IDs
    products = []
    for cat, nouns in CATEGORIES.items():
        for i, noun in enumerate(nouns):
            for variant in range(4):
                products.append((f"P{len(products) + 1:04d}", cat, noun))
    prod_df = pd.DataFrame(products, columns=["product_id", "category", "product_name"])

    rows = []
    cat_names = list(CATEGORIES)
    for i, d in enumerate(sorted(review_dates)):
        d = pd.Timestamp(d)
        cat = rng.choices(cat_names, weights=CATEGORY_WEIGHTS)[0]
        prod = prod_df[prod_df.category == cat].sample(1, random_state=nrng.integers(1e9)).iloc[0]
        text, rating, aspects = build_review(rng, nrng, cat, prod.product_name, d)
        rows.append({
            "review_id": f"R{i + 1:05d}",
            "review_date": d.date().isoformat(),
            "product_id": prod.product_id,
            "product_name": prod.product_name.title(),
            "category": cat,
            "region": rng.choice(REGIONS),
            "verified_purchase": rng.random() < 0.88,
            "rating": rating,
            "helpful_votes": int(nrng.poisson(1.2 + (2.5 if rating <= 2 else 0))),
            "review_text": text,
        })

    df = pd.DataFrame(rows)

    # Realistic data-quality issues for the cleaning step
    dup_idx = nrng.choice(df.index, size=25, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)          # duplicate submissions
    blank_idx = nrng.choice(df.index, size=18, replace=False)
    df.loc[blank_idx, "review_text"] = ""                               # empty reviews
    na_idx = nrng.choice(df.index, size=30, replace=False)
    df.loc[na_idx, "region"] = None                                     # missing region
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
