"""
brain/prompt_parser.py  (v8 — complete rewrite)
================================================
Pure-Python intent classifier + keyword extractor.
Zero external dependencies. Python 3.9-3.14+.

ROOT CAUSE FIXED:
  - Intent classification was not being passed to the UI as a clean structured field.
  - Keyword generation was producing overly specific multi-word phrases
    (e.g. "housing price") that Kaggle's API does not recognise.
  - Fallback dataset names (known Kaggle slugs) are now attached to the spec
    so the collector can use them when generic search returns 0 results.

DESIGN:
  Intent  → ml_model | chatbot
  Task    → regression | classification | clustering | chatbot
  Domain  → the subject area (e.g. "housing", "medical", "finance")
  Keywords→ short 1-word terms proven to work with Kaggle/UCI/OpenML
  Fallback datasets → list of known dataset refs for the detected domain
"""
from __future__ import annotations
import re
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ── Task-type signals ──────────────────────────────────────────────────────────
_REGRESSION_SIGNALS = {
    "predict", "price", "cost", "salary", "revenue", "forecast", "estimate",
    "value", "rent", "score", "rate", "amount", "sales", "demand", "profit",
    "temperature", "weight", "height", "income", "expense", "numeric",
    "continuous", "quantity", "how much", "how many", "regression",
}
_CLASSIFICATION_SIGNALS = {
    "classify", "classification", "detect", "detection", "spam", "fraud",
    "cancer", "disease", "sentiment", "category", "churn", "diagnosis",
    "binary", "multiclass", "label", "whether", "will it", "is it",
    "true or false", "yes or no", "positive negative", "approve", "reject",
}
_CLUSTERING_SIGNALS = {
    "cluster", "segment", "group", "similar", "similarity", "recommend",
    "recommendation", "suggest", "collaborative", "filter", "nearest",
    "unsupervised", "k-means", "topic", "community",
}
_CHATBOT_SIGNALS = {
    "chatbot", "chat", "qa", "faq", "assistant", "knowledge base",
    "retrieval", "rag", "summarize", "summarise", "question answer",
    "question answering", "document qa", "conversational",
}

# ── Noise words (never appear in search queries) ──────────────────────────────
_NOISE = {
    "build", "create", "generate", "make", "want", "need", "use", "using",
    "like", "give", "get", "let", "please", "write", "production", "senior",
    "engineer", "code", "goal", "system", "application", "app", "model",
    "based", "given", "input", "output", "data", "dataset", "train", "test",
    "feature", "column", "machine", "learning", "deep", "neural", "network",
    "artificial", "intelligence", "algorithm", "pipeline", "workflow",
    "project", "develop", "implement", "design", "solution", "platform",
    "tool", "framework", "library", "module", "package", "service",
    "api", "backend", "frontend", "interface", "user", "client", "server",
    "predict", "classification", "regression", "clustering", "detection",
    "analysis", "analytics", "perform", "perform", "accurate", "accurate",
    "simple", "complex", "robust", "efficient", "ready", "real", "world",
}

_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","do","does",
    "did","will","would","could","should","may","might","can","i","me","my",
    "we","our","you","your","he","she","it","its","they","their","this",
    "that","these","those","what","which","who","how","when","where","why",
    "all","each","every","some","no","not","only","so","than","very","just",
    "also","any","about","into","than","both","each","here","there","then",
}

# ── Domain map: token → (search_keywords, fallback_kaggle_refs) ──────────────
# fallback_kaggle_refs are REAL Kaggle dataset slugs that WORK when search fails
_DOMAIN_MAP: dict[str, dict] = {
    # Housing / Real Estate
    "house":      {"kw": ["house prices", "housing"],
                   "fb": ["harlfoxem/house-prices-dataset",
                          "camnugent/california-housing-prices",
                          "yasserh/housing-prices-dataset"]},
    "housing":    {"kw": ["housing", "house prices"],
                   "fb": ["camnugent/california-housing-prices",
                          "harlfoxem/house-prices-dataset"]},
    "bedroom":    {"kw": ["house prices", "real estate"],
                   "fb": ["harlfoxem/house-prices-dataset"]},
    "bedrooms":   {"kw": ["house prices", "real estate"],
                   "fb": ["harlfoxem/house-prices-dataset"]},
    "real":       {"kw": ["real estate"], "fb": []},
    "estate":     {"kw": ["real estate", "housing"],
                   "fb": ["harlfoxem/house-prices-dataset"]},
    "apartment":  {"kw": ["apartment", "housing"],
                   "fb": ["arnavkulkarni1998/apartments-data"]},
    "rent":       {"kw": ["rent", "rental"],
                   "fb": ["dgomonov/new-york-city-airbnb-open-data"]},

    # Movies / Entertainment
    "movie":      {"kw": ["movies", "film"],
                   "fb": ["rounakbanik/the-movies-dataset",
                          "tmdb/tmdb-movie-metadata",
                          "netflix-inc/netflix-shows"]},
    "film":       {"kw": ["movies", "film"],
                   "fb": ["rounakbanik/the-movies-dataset"]},
    "movies":     {"kw": ["movies"], "fb": ["rounakbanik/the-movies-dataset"]},
    "rating":     {"kw": ["ratings"],
                   "fb": ["rounakbanik/the-movies-dataset",
                          "grouplens/movielens-20m-dataset"]},
    "ratings":    {"kw": ["ratings"], "fb": ["grouplens/movielens-20m-dataset"]},
    "genre":      {"kw": ["movies", "genre"],
                   "fb": ["rounakbanik/the-movies-dataset"]},
    "recommendation": {"kw": ["movies", "ratings"],
                   "fb": ["grouplens/movielens-20m-dataset",
                          "rounakbanik/the-movies-dataset"]},
    "recommend":  {"kw": ["movies", "ratings"],
                   "fb": ["grouplens/movielens-20m-dataset"]},

    # Finance / Stock / Credit
    "stock":      {"kw": ["stocks", "stock market"],
                   "fb": ["jacksoncrow/nse-and-bse-stocks-with-fundamentals",
                          "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"]},
    "finance":    {"kw": ["finance", "financial"],
                   "fb": ["mlg-ulb/creditcardfraud"]},
    "credit":     {"kw": ["credit", "loan"],
                   "fb": ["mlg-ulb/creditcardfraud",
                          "uciml/default-of-credit-card-clients-dataset"]},
    "fraud":      {"kw": ["fraud", "credit card"],
                   "fb": ["mlg-ulb/creditcardfraud"]},
    "loan":       {"kw": ["loan", "credit"],
                   "fb": ["wordsforthewise/lending-club",
                          "uciml/default-of-credit-card-clients-dataset"]},
    "salary":     {"kw": ["salary", "wages"],
                   "fb": ["kaggle/sf-salaries",
                          "ryanxjhan/us-salaries-1990-2016"]},
    "income":     {"kw": ["income", "salary"],
                   "fb": ["uciml/adult-census-income"]},

    # Medical / Health
    "diabetes":   {"kw": ["diabetes"],
                   "fb": ["uciml/pima-indians-diabetes-database",
                          "mathchi5/diabetes-data-set"]},
    "heart":      {"kw": ["heart disease"],
                   "fb": ["ronitf/heart-disease-uci",
                          "cherngs/heart-disease-cleveland-uci"]},
    "cancer":     {"kw": ["cancer"],
                   "fb": ["uciml/breast-cancer-wisconsin-data",
                          "erdemtaha/cancer-data"]},
    "medical":    {"kw": ["medical", "health"],
                   "fb": ["uciml/pima-indians-diabetes-database"]},
    "health":     {"kw": ["health", "medical"],
                   "fb": ["uciml/pima-indians-diabetes-database"]},
    "patient":    {"kw": ["medical", "patient"],
                   "fb": ["uciml/pima-indians-diabetes-database"]},
    "disease":    {"kw": ["disease", "medical"],
                   "fb": ["ronitf/heart-disease-uci"]},

    # E-commerce / Customer
    "customer":   {"kw": ["customer", "ecommerce"],
                   "fb": ["olistbr/brazilian-ecommerce",
                          "blastchar/telco-customer-churn"]},
    "churn":      {"kw": ["churn", "customer churn"],
                   "fb": ["blastchar/telco-customer-churn",
                          "becksddf/churn-modelling"]},
    "sales":      {"kw": ["sales"],
                   "fb": ["manjeetsingh/retaildataset",
                          "rohitsahoo/sales-forecasting"]},
    "ecommerce":  {"kw": ["ecommerce", "retail"],
                   "fb": ["olistbr/brazilian-ecommerce"]},
    "retail":     {"kw": ["retail", "sales"],
                   "fb": ["manjeetsingh/retaildataset"]},
    "product":    {"kw": ["products", "ecommerce"],
                   "fb": ["olistbr/brazilian-ecommerce"]},

    # Sentiment / Text / NLP
    "sentiment":  {"kw": ["sentiment", "reviews"],
                   "fb": ["marklvl/sentiment-analysis-dataset",
                          "snap/amazon-fine-food-reviews"]},
    "review":     {"kw": ["reviews", "sentiment"],
                   "fb": ["snap/amazon-fine-food-reviews"]},
    "reviews":    {"kw": ["reviews"],
                   "fb": ["snap/amazon-fine-food-reviews"]},
    "spam":       {"kw": ["spam", "email"],
                   "fb": ["uciml/sms-spam-collection-dataset"]},
    "tweet":      {"kw": ["twitter", "tweets"],
                   "fb": ["kazanova/sentiment140"]},
    "twitter":    {"kw": ["twitter"],
                   "fb": ["kazanova/sentiment140"]},

    # Weather / Environment
    "weather":    {"kw": ["weather", "climate"],
                   "fb": ["muthuj/weather-dataset",
                          "jsphyg/weather-dataset-rattle-package"]},
    "temperature":{"kw": ["temperature", "weather"],
                   "fb": ["muthuj/weather-dataset"]},
    "climate":    {"kw": ["climate", "weather"],
                   "fb": ["berkeleyearth/climate-change-earth-surface-temperature-data"]},

    # Transport
    "taxi":       {"kw": ["taxi", "trips"],
                   "fb": ["elemento/nyc-yellow-taxi-trip-data"]},
    "flight":     {"kw": ["flights", "airline"],
                   "fb": ["usdot/flight-delays",
                          "open-flights/flight-route-database"]},
    "airline":    {"kw": ["airline", "flights"],
                   "fb": ["usdot/flight-delays"]},
    "uber":       {"kw": ["uber", "taxi"],
                   "fb": ["fivethirtyeight/uber-pickups-in-new-york-city"]},
    "traffic":    {"kw": ["traffic"],
                   "fb": ["fedesoriano/traffic-volume-data-set"]},

    # Education
    "student":    {"kw": ["student", "education"],
                   "fb": ["uciml/student-performance",
                          "devansodariya/student-performance-data-set"]},
    "education":  {"kw": ["education", "student"],
                   "fb": ["uciml/student-performance"]},
    "academic":   {"kw": ["academic", "student"],
                   "fb": ["uciml/student-performance"]},

    # Energy
    "energy":     {"kw": ["energy", "electricity"],
                   "fb": ["uciml/electric-power-consumption",
                          "nicholasjhana/energy-consumption-generation-prices-and-weather"]},
    "electricity":{"kw": ["electricity", "energy"],
                   "fb": ["uciml/electric-power-consumption"]},
    "power":      {"kw": ["power", "energy"],
                   "fb": ["uciml/electric-power-consumption"]},

    # Classic benchmarks
    "titanic":    {"kw": ["titanic"],
                   "fb": ["heptapod/titanic"]},
    "iris":       {"kw": ["iris"],
                   "fb": ["uciml/iris"]},
    "wine":       {"kw": ["wine"],
                   "fb": ["uciml/wine-quality"]},
    "mnist":      {"kw": ["digit", "handwritten"],
                   "fb": ["hojjatk/mnist-dataset"]},
    "boston":     {"kw": ["boston housing", "house prices"],
                   "fb": ["harlfoxem/house-prices-dataset"]},
    "california": {"kw": ["california housing"],
                   "fb": ["camnugent/california-housing-prices"]},

    # HR / Employment
    "employee":   {"kw": ["employee", "hr"],
                   "fb": ["pavansubhasht/ibm-hr-analytics-attrition-dataset"]},
    "attrition":  {"kw": ["attrition", "hr"],
                   "fb": ["pavansubhasht/ibm-hr-analytics-attrition-dataset"]},
    "hiring":     {"kw": ["hr", "hiring"],
                   "fb": ["pavansubhasht/ibm-hr-analytics-attrition-dataset"]},

    # Agriculture / Biology
    "plant":      {"kw": ["plant", "species"],
                   "fb": ["uciml/iris"]},
    "crop":       {"kw": ["crop", "agriculture"],
                   "fb": ["atharvaingle/crop-recommendation-dataset"]},
    "agriculture":{"kw": ["agriculture", "crop"],
                   "fb": ["atharvaingle/crop-recommendation-dataset"]},

    # Insurance / Risk
    "insurance":  {"kw": ["insurance"],
                   "fb": ["mirichoi0218/insurance"]},
    "risk":       {"kw": ["risk", "insurance"],
                   "fb": ["mirichoi0218/insurance"]},
}

# Universal fallback when nothing else works — always available on Kaggle
_UNIVERSAL_FALLBACKS = {
    "regression":     ["harlfoxem/house-prices-dataset",
                       "camnugent/california-housing-prices",
                       "mirichoi0218/insurance"],
    "classification": ["uciml/pima-indians-diabetes-database",
                       "blastchar/telco-customer-churn",
                       "mlg-ulb/creditcardfraud"],
    "clustering":     ["grouplens/movielens-20m-dataset",
                       "rounakbanik/the-movies-dataset"],
    "chatbot":        ["harlfoxem/house-prices-dataset"],
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t]


def _is_useful(token: str) -> bool:
    return (
        len(token) >= 3
        and token not in _STOP
        and token not in _NOISE
        and not token.isdigit()
        and bool(re.match(r"^[a-z][a-z0-9_\-]*$", token))
    )


def _build_spec_from_prompt(prompt: str) -> dict:
    """
    Full NLP analysis of the prompt.
    Returns search keywords, fallback dataset refs, and task classification.
    """
    tokens    = _tokenize(prompt)
    low       = prompt.lower()

    # ── Task classification ──────────────────────────────────────────────────
    reg_score   = sum(1 for w in _REGRESSION_SIGNALS    if w in low)
    cls_score   = sum(1 for w in _CLASSIFICATION_SIGNALS if w in low)
    clust_score = sum(1 for w in _CLUSTERING_SIGNALS     if w in low)
    chat_score  = sum(1 for w in _CHATBOT_SIGNALS        if w in low)

    if chat_score > max(reg_score, cls_score, clust_score):
        intent    = "chatbot"
        task_type = "chatbot"
    elif clust_score > max(reg_score, cls_score):
        intent    = "ml_model"
        task_type = "clustering"
    elif cls_score > reg_score:
        intent    = "ml_model"
        task_type = "classification"
    else:
        intent    = "ml_model"
        task_type = "regression"

    # ── Domain detection & keyword extraction ────────────────────────────────
    search_keywords: list[str] = []
    fallback_refs:   list[str] = []
    domain_tokens:   list[str] = []
    seen_kw:         set[str]  = set()

    for tok in tokens:
        if tok in _DOMAIN_MAP:
            domain_tokens.append(tok)
            entry = _DOMAIN_MAP[tok]
            for kw in entry["kw"]:
                if kw not in seen_kw:
                    seen_kw.add(kw)
                    search_keywords.append(kw)
            for ref in entry["fb"]:
                if ref not in fallback_refs:
                    fallback_refs.append(ref)

    # If no domain keywords found, add useful raw tokens as search terms
    if not search_keywords:
        for tok in tokens:
            if _is_useful(tok) and tok not in seen_kw:
                seen_kw.add(tok)
                search_keywords.append(tok)

    # Add task-type universal fallbacks if domain gave none
    if not fallback_refs:
        fallback_refs = _UNIVERSAL_FALLBACKS.get(task_type, [])

    domain = " ".join(domain_tokens[:2]) if domain_tokens else (
        search_keywords[0] if search_keywords else "general"
    )

    # Final dedup and truncation
    search_keywords = list(dict.fromkeys(search_keywords))[:6]
    fallback_refs   = list(dict.fromkeys(fallback_refs))[:5]

    # ── Input params ─────────────────────────────────────────────────────────
    input_params = _extract_input_params(low, tokens)

    # ── Target param ─────────────────────────────────────────────────────────
    target_param = _extract_target(low, task_type)

    return {
        "raw":            prompt,
        "intent":         intent,
        "task_type":      task_type,
        "domain":         domain,
        "keywords":       search_keywords,
        "fallback_refs":  fallback_refs,
        "input_params":   input_params,
        "target_param":   target_param,
        # Scores for transparency
        "_scores": {
            "regression":     reg_score,
            "classification": cls_score,
            "clustering":     clust_score,
            "chatbot":        chat_score,
        },
    }


def _extract_input_params(low: str, tokens: list[str]) -> list[str]:
    params: list[str] = []
    pattern = re.compile(
        r"\b(based on|using|given|for a|with|input|provide|enter)\b"
        r"\s+([a-z0-9 ,/&]+?)(?:\s+and\s+([a-z0-9 ,/&]+?))?(?:\s*,|\s+to\s|\s+predict|\s+we\s|$)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(low):
        raw = " ".join(filter(None, [m.group(2), m.group(3)]))
        for p in re.split(r"[,/&]|\band\b|\bor\b", raw):
            p = p.strip()
            if p and len(p) > 2 and p not in _STOP and p not in _NOISE:
                params.append(p)
    if not params:
        for tok in tokens:
            if _is_useful(tok):
                params.append(tok)
    seen: set[str] = set()
    return [p for p in params if not (p in seen or seen.add(p))][:8]  # type: ignore[func-returns-value]


def _extract_target(low: str, task_type: str) -> str:
    m = re.search(
        r"\b(predict|estimate|forecast|output|determine|classify)\s+([a-z0-9_ ]+)", low
    )
    if m:
        cand = m.group(2).strip().split()[0]
        if cand not in _NOISE and cand not in _STOP:
            return cand
    return {"regression": "price", "classification": "label",
            "clustering": "cluster", "chatbot": "response"}.get(task_type, "output")


class PromptParser:
    """
    Pure-Python intent classifier and keyword extractor.
    No spaCy, no NLTK. Works on Python 3.9-3.14+.
    """

    def parse(self, prompt: str) -> dict:
        result = _build_spec_from_prompt(prompt)
        logger.info(
            "Parsed prompt → intent=%s  task=%s  domain=%s  keywords=%s  "
            "fallback_refs=%d  inputs=%s  target=%s  scores=%s",
            result["intent"], result["task_type"], result["domain"],
            result["keywords"], len(result["fallback_refs"]),
            result["input_params"], result["target_param"],
            result["_scores"],
        )
        return result
