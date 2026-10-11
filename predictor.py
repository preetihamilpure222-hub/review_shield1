"""
ReviewShield AI - analysis engine.

Works on reviews from ANY platform (Amazon, Flipkart, Myntra, Google, ...).
It needs only review text; rating, date, product and verified-purchase
columns are optional and unlock extra analytics.

Models used
-----------
1. Fake-review detector   : TF-IDF + Multinomial Naive Bayes (your v2 model)
2. Sentiment / star model : TF-IDF + Logistic Regression + Ridge (trained
                            on the Amazon dataset, models/sentiment_rating_model.pkl)
3. Rule-based red flags   : duplicates, shouting, promo phrases, rating/text mismatch
"""

import os
import re
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from features import feats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE_DIR, "models")

# How much each signal contributes to the final risk score.
ML_WEIGHT = 0.70
RULE_WEIGHT = 0.30
SUSPICIOUS_AT = 0.50
UNCERTAIN_AT = 0.40

MAX_ROWS = 10000

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
fake_model = joblib.load(os.path.join(MODELS, "fake_review_model_v2.pkl"))
fake_vectorizer = joblib.load(os.path.join(MODELS, "tfidf_vectorizer_v2.pkl"))

_v3 = joblib.load(os.path.join(MODELS, "fake_review_model_v3.pkl"))
FAKE_METRICS = {"text": _v3["text"]["metrics"], "meta": _v3["meta"]["metrics"]}

_sent = joblib.load(os.path.join(MODELS, "sentiment_rating_model.pkl"))
sent_vectorizer = _sent["vectorizer"]
sent_classifier = _sent["classifier"]
rating_regressor = _sent["regressor"]
SENTIMENT_METRICS = _sent["metrics"]


# --------------------------------------------------------------------------
# Text cleaning  (identical to the training notebook)
# --------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Rule-based red flags
# --------------------------------------------------------------------------
PROMO_PHRASES = [
    "must buy", "best product ever", "highly recommend to everyone",
    "everyone should buy", "buy it now", "buy now", "100% genuine",
    "life changing", "worth every penny", "five stars", "5 stars",
    "go for it", "don't think twice", "dont think twice", "paisa vasool",
    "totally satisfied", "awesome product", "superb product",
]


# Product acronyms are not "shouting" - ignore them in the ALL-CAPS rule.
ACRONYMS = {
    "USB", "HDMI", "LED", "LCD", "OLED", "AMOLED", "GPS", "RAM", "ROM", "SSD",
    "HDD", "CPU", "GPU", "NFC", "SIM", "OTG", "VGA", "DVI", "SD", "TV", "PC",
    "AC", "DC", "UV", "IPS", "FHD", "UHD", "HD", "TWS", "ANC", "RGB", "AI",
    "OS", "UI", "APP", "WIFI", "LAN", "RJ", "MAH", "PD", "QC", "EMI", "ISO",
    "PUBG", "COD", "BGMI", "OK", "AM", "PM", "US", "UK", "EU", "FAQ", "DSLR",
}


def caps_word_count(words):
    """Count genuinely shouted words (skips acronyms and anything with digits)."""
    n = 0
    for w in words:
        t = w.strip(".,!?;:()[]\"'-")
        if (len(t) > 2 and t.isupper() and t.isalpha() and t not in ACRONYMS):
            n += 1
    return n


def rule_flags(raw, rating=None, sentiment=None, duplicate=False):
    """Return (score 0-1, [flag descriptions])."""
    text = str(raw)
    words = text.split()
    low = text.lower()
    flags, score = [], 0.0

    if duplicate:
        flags.append("Identical text appears in other reviews")
        score += 0.50
    if re.search(r"https?://|www\.", low):
        flags.append("Contains an external link")
        score += 0.20
    if text.count("!") >= 3:
        flags.append("Excessive exclamation marks")
        score += 0.20
    if caps_word_count(words) >= 3:
        flags.append("Several ALL-CAPS words")
        score += 0.15
    if any(p in low for p in PROMO_PHRASES):
        flags.append("Promotional / pushy phrasing")
        score += 0.15
    if 0 < len(words) < 4:
        flags.append("Very short, low-information review")
        score += 0.10
    if rating is not None and sentiment is not None:
        if rating >= 4 and sentiment == "negative":
            flags.append("High rating but negative wording")
            score += 0.20
        elif rating <= 2 and sentiment == "positive":
            flags.append("Low rating but positive wording")
            score += 0.20
    return min(score, 1.0), flags


# --------------------------------------------------------------------------
# Core scoring
# --------------------------------------------------------------------------
def _stack_prob(bundle, cleaned, raw, ratings=None, verified=None):
    """Two-stage model: TF-IDF LR probability -> gradient boosting with style (+meta) features."""
    p = bundle["lr"].predict_proba(bundle["vectorizer"].transform(cleaned))[:, 1]
    cols = [np.array([feats(t) for t in raw]), p]
    if bundle["meta"]:
        cols += [np.asarray(ratings, dtype=float)[:, None], np.asarray(verified, dtype=float)[:, None]]
    return bundle["gb"].predict_proba(np.column_stack(cols))[:, 1]


def fake_probabilities(raw, cleaned, ratings, verified):
    """Use the text+rating+verified model when metadata exists, else the text-only model."""
    n = len(raw)
    out = _stack_prob(_v3["text"], cleaned, raw)
    if ratings is not None and verified is not None:
        r = np.array([np.nan if (x is None or pd.isna(x)) else float(x) for x in ratings])
        v = np.array([np.nan if (x is None or pd.isna(x)) else float(x) for x in verified])
        ok = ~np.isnan(r) & ~np.isnan(v)
        if ok.any():
            idx = np.where(ok)[0]
            out[idx] = _stack_prob(_v3["meta"], [cleaned[i] for i in idx], [raw[i] for i in idx], r[idx], v[idx])
    return out


def verdict_for(risk):
    if risk >= SUSPICIOUS_AT:
        return "Suspicious"
    if risk >= UNCERTAIN_AT:
        return "Uncertain"
    return "Genuine"


def score_texts(raw_texts, ratings=None, verified=None):
    """Score a list of raw review strings. Returns list of dicts."""
    n = len(raw_texts)
    cleaned = [clean_text(t) for t in raw_texts]

    fake_p = fake_probabilities(raw_texts, cleaned, ratings, verified)

    sv = sent_vectorizer.transform(cleaned)
    sentiments = sent_classifier.predict(sv)
    star_pred = np.clip(rating_regressor.predict(sv), 1, 5)

    counts = Counter(cleaned)
    out = []
    for i in range(n):
        dup = bool(cleaned[i]) and counts[cleaned[i]] > 1
        r = ratings[i] if ratings is not None else None
        r = None if (r is None or pd.isna(r)) else float(r)
        rs, flags = rule_flags(raw_texts[i], r, sentiments[i], dup)
        risk = ML_WEIGHT * float(fake_p[i]) + RULE_WEIGHT * rs
        out.append({
            "ml_fake_prob": round(float(fake_p[i]) * 100, 1),
            "rule_score": round(rs * 100, 1),
            "risk": round(risk * 100, 1),
            "verdict": verdict_for(risk),
            "sentiment": str(sentiments[i]),
            "predicted_stars": round(float(star_pred[i]), 1),
            "flags": flags,
            "words": len(cleaned[i].split()),
        })
    return out


def explain_fake_words(raw, top_n=6):
    """Words pushing the NB model toward 'fake' (+) or 'genuine' (-)."""
    vec = fake_vectorizer.transform([clean_text(raw)]).tocsr()
    if vec.nnz == 0:
        return {"fake": [], "genuine": []}
    names = fake_vectorizer.get_feature_names_out()
    diff = fake_model.feature_log_prob_[1] - fake_model.feature_log_prob_[0]
    items = [(names[j], float(v * diff[j])) for j, v in zip(vec.indices, vec.data)]
    items.sort(key=lambda x: x[1], reverse=True)
    return {
        "fake": [w for w, s in items if s > 0][:top_n],
        "genuine": [w for w, s in reversed(items) if s < 0][:top_n],
    }


# --------------------------------------------------------------------------
# File loading and column detection (Amazon / Flipkart / generic)
# --------------------------------------------------------------------------
ALIASES = {
    "text": ["review_text", "review text", "reviewtext", "review", "reviews",
             "text", "content", "comment", "comments", "body", "review_body",
             "review body", "customer review", "description", "feedback"],
    "title": ["review_title", "review title", "title", "summary", "headline",
              "review_headline"],
    "rating": ["rating", "ratings", "stars", "star", "star_rating", "score",
               "review_rating", "overall", "rating (out of 5)"],
    "date": ["date", "review_date", "reviewed_on", "review date", "time",
             "reviewtime", "created_at", "timestamp", "posted_on",
             "reviewed on"],
    "product": ["product", "product_name", "product_title", "product name",
                "asin", "product_id", "productid", "item", "name"],
    "verified": ["verified_purchase", "verified purchase", "verified",
                 "certified buyer", "certified_buyer"],
}


def load_table(file_storage):
    """Read an uploaded CSV / TSV / TXT / XLSX / JSON into a DataFrame."""
    name = (file_storage.filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_storage)
    elif name.endswith(".json"):
        df = pd.read_json(file_storage)
    elif name.endswith((".tsv", ".txt")):
        df = pd.read_csv(file_storage, sep=None, engine="python",
                         on_bad_lines="skip")
    elif name.endswith(".csv"):
        df = pd.read_csv(file_storage, encoding_errors="replace",
                         on_bad_lines="skip")
    else:
        raise ValueError("Unsupported file type. Use CSV, XLSX, TSV, TXT or JSON.")
    if df.empty:
        raise ValueError("The file contains no rows.")
    return df


def text_to_table(raw, mode="auto"):
    """Pasted text -> one row per review.

    mode: "blank" = reviews separated by blank lines,
          "line"  = one review per line,
          "single"= the whole text is ONE review (keeps its line breaks),
          "auto"  = blank-line blocks if present; else one-per-line when most
                    lines are sentence-length; else treat everything as one review.
    """
    raw = raw.strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if mode == "single":
        out = [re.sub(r"\s*\n\s*", " ", raw)]
    elif mode == "line":
        out = lines
    elif mode == "blank":
        out = [re.sub(r"\s*\n\s*", " ", b) for b in blocks]
    else:  # auto
        if len(blocks) >= 2:
            out = [re.sub(r"\s*\n\s*", " ", b) for b in blocks]
        elif len(lines) >= 2 and sum(len(l) >= 40 for l in lines) / len(lines) > 0.7 \
                and not any(re.match(r"^(\d+[.)]|[-*\u2022])\s", l) for l in lines):
            out = lines
        else:
            out = [re.sub(r"\s*\n\s*", " ", raw)]
    return pd.DataFrame({"review": out})


def detect_columns(df):
    """Map logical fields -> actual column names using aliases."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for field, names in ALIASES.items():
        for a in names:
            if a in lowered and lowered[a] not in mapping.values():
                mapping[field] = lowered[a]
                break
    # Fallback: the column with the longest average text is probably the review.
    if "text" not in mapping:
        best, best_len = None, 0
        for c in df.columns:
            if df[c].dtype == object:
                avg = df[c].astype(str).str.len().mean()
                if avg > best_len:
                    best, best_len = c, avg
        if best is not None and best_len > 20:
            mapping["text"] = best
    return mapping


def _parse_rating(series):
    # Handles 4, 4.0, "4.0 out of 5 stars", "4 stars"
    s = series.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 1) & (s <= 5))


# --------------------------------------------------------------------------
# Dataset-level analytics + prediction
# --------------------------------------------------------------------------
STOP = set(ENGLISH_STOP_WORDS) | {
    "product", "item", "just", "really", "bought", "buy", "got", "ordered",
    "amazon", "flipkart", "use", "using", "used", "br", "like", "good",
    "great", "nice", "one", "will", "would", "also", "get", "very",
}


def top_terms(texts, n=12):
    c = Counter()
    for t in texts:
        toks = re.findall(r"[a-z]{3,}", clean_text(t))
        c.update(w for w in set(toks) if w not in STOP)
    return [{"term": w, "count": k} for w, k in c.most_common(n)]


def forecast_series(dates, values, periods=3):
    """Monthly mean + linear-trend forecast. Returns None if too little data."""
    s = pd.DataFrame({"d": dates, "v": values}).dropna()
    if len(s) < 8:
        return None
    s["m"] = s["d"].dt.to_period("M")
    g = s.groupby("m")["v"].agg(["mean", "count"])
    if len(g) < 4:
        return None
    x = np.arange(len(g))
    slope, intercept = np.polyfit(x, g["mean"].values, 1)
    fx = np.arange(len(g), len(g) + periods)
    resid = g["mean"].values - (slope * x + intercept)
    sd = float(np.std(resid)) if len(resid) > 2 else 0.0
    last = g.index[-1]
    return {
        "labels": [str(p) for p in g.index],
        "actual": [round(float(v), 3) for v in g["mean"].values],
        "counts": [int(v) for v in g["count"].values],
        "forecast_labels": [str(last + (i + 1)) for i in range(periods)],
        "forecast": [round(float(slope * i + intercept), 3) for i in fx],
        "band": round(1.28 * sd, 3),
        "slope_per_month": round(float(slope), 4),
    }


def analyze_dataframe(df, mapping, platform="Other"):
    """Run the full pipeline. Returns (results list, summary dict)."""
    if "text" not in mapping:
        raise ValueError("Could not find the review-text column. "
                         "Please choose it manually.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"File has {len(df):,} rows; the limit is {MAX_ROWS:,}.")

    work = pd.DataFrame({"text": df[mapping["text"]].fillna("").astype(str)})
    if "title" in mapping:
        t = df[mapping["title"]].fillna("").astype(str)
        work["text"] = (t + ". " + work["text"]).str.strip(". ")
    work["rating"] = _parse_rating(df[mapping["rating"]]) if "rating" in mapping else np.nan
    work["date"] = (pd.to_datetime(df[mapping["date"]], errors="coerce")
                    if "date" in mapping else pd.NaT)
    work["product"] = df[mapping["product"]].astype(str) if "product" in mapping else ""
    use_verified = False
    if "verified" in mapping:
        v = df[mapping["verified"]].astype(str).str.strip().str.lower()
        # only trust the column when it holds real values (a column of None/blank means "unknown")
        use_verified = bool((df[mapping["verified"]].notna() & ~v.isin(["", "none", "nan", "null", "nat"])).any())
    if use_verified:
        work["verified"] = v.isin(["y", "yes", "true", "1", "verified purchase", "certified buyer"])
    else:
        work["verified"] = np.nan

    work = work[work["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if work.empty:
        raise ValueError("No non-empty reviews found in the selected column.")

    scored = score_texts(work["text"].tolist(), work["rating"].tolist(),
                         work["verified"].tolist() if use_verified else None)

    results = []
    for i, s in enumerate(scored):
        row = {"row": i + 1, "review": work.at[i, "text"][:600], **s}
        r = work.at[i, "rating"]
        row["rating"] = None if pd.isna(r) else float(r)
        d = work.at[i, "date"]
        row["date"] = None if pd.isna(d) else d.strftime("%Y-%m-%d")
        row["product"] = work.at[i, "product"][:80]
        v = work.at[i, "verified"]
        row["verified"] = None if pd.isna(v) else bool(v)
        results.append(row)

    res = pd.DataFrame(results)
    total = len(res)
    verdicts = res["verdict"].value_counts().to_dict()
    n_susp = int(verdicts.get("Suspicious", 0))
    n_unc = int(verdicts.get("Uncertain", 0))
    n_gen = int(verdicts.get("Genuine", 0))

    has_rating = bool(res["rating"].notna().any())
    raw_avg = float(res["rating"].mean()) if has_rating else None
    clean_avg = (float(res.loc[res["verdict"] != "Suspicious", "rating"].mean())
                 if has_rating and (n_gen + n_unc) > 0 else None)

    sentiment_counts = res["sentiment"].value_counts().to_dict()
    rating_dist, susp_by_rating = {}, {}
    if has_rating:
        tmp = res.dropna(subset=["rating"]).copy()
        tmp["star"] = tmp["rating"].round().astype(int)
        rating_dist = {str(k): int(v) for k, v in tmp["star"].value_counts().sort_index().items()}
        for star, g in tmp.groupby("star"):
            susp_by_rating[str(star)] = round(float((g["verdict"] == "Suspicious").mean() * 100), 1)

    hist, _ = np.histogram(res["risk"], bins=10, range=(0, 100))

    forecast = None
    if res["date"].notna().sum() >= 8:
        dts = pd.to_datetime(res["date"])
        base = res["rating"] if has_rating else res["predicted_stars"]
        rating_fc = forecast_series(dts, base.values)
        susp_fc = forecast_series(dts, (res["verdict"] == "Suspicious").astype(float).values * 100)
        if rating_fc:
            forecast = {
                "metric": "Average rating" if has_rating else "Predicted rating",
                "rating": rating_fc,
                "suspicious": susp_fc,
            }

    pos_texts = res.loc[res["sentiment"] == "positive", "review"].tolist()
    neg_texts = res.loc[res["sentiment"] == "negative", "review"].tolist()

    summary = {
        "platform": platform,
        "total": total,
        "suspicious": n_susp,
        "uncertain": n_unc,
        "genuine": n_gen,
        "suspicious_pct": round(n_susp / total * 100, 1),
        "trust_score": round(100 - float(res["risk"].mean()), 1),
        "avg_risk": round(float(res["risk"].mean()), 1),
        "raw_avg_rating": round(raw_avg, 2) if raw_avg is not None else None,
        "trusted_avg_rating": round(clean_avg, 2) if clean_avg is not None else None,
        "avg_predicted_stars": round(float(res["predicted_stars"].mean()), 2),
        "avg_words": round(float(res["words"].mean()), 1),
        "duplicates": int(sum(any("Identical text" in f for f in fl) for fl in res["flags"])),
        "sentiment": {k: int(sentiment_counts.get(k, 0))
                      for k in ("positive", "neutral", "negative")},
        "rating_dist": rating_dist,
        "susp_by_rating": susp_by_rating,
        "risk_hist": [int(x) for x in hist],
        "top_positive": top_terms(pos_texts),
        "top_negative": top_terms(neg_texts),
        "flag_counts": dict(Counter(f for fl in res["flags"] for f in fl).most_common(8)),
        "forecast": forecast,
        "columns_used": {k: str(v) for k, v in mapping.items()},
        "has_rating": has_rating,
        "has_dates": forecast is not None,
    }
    return results, summary


def analyze_single(review, rating=None, verified=None):
    s = score_texts([review], [rating] if rating else None,
                    [1.0 if verified else 0.0] if verified is not None else None)[0]
    s["model_used"] = "text+rating+verified" if (rating and verified is not None) else "text-only"
    s["explanation"] = explain_fake_words(review)
    text = review.strip()
    w = text.split()
    s["stats"] = {
        "characters": len(text),
        "words": len(w),
        "sentences": len([x for x in re.split(r"[.!?]+", text) if x.strip()]),
        "avg_word_len": round(sum(len(x.strip('.,!?;:')) for x in w) / len(w), 2) if w else 0,
        "exclamations": text.count("!"),
        "caps_words": caps_word_count(w),
    }
    return s
