import ast
import json
import os
import pickle
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "522ee874f1ad7fda68394ed5805a2f79")
BASE_URL = "https://api.themoviedb.org/3"

# --- 1. PAGE SETUP & DESIGN ---
st.set_page_config(page_title="Casting Sandbox", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-0: #0b0f13;
        --bg-1: #141a20;
        --card: rgba(26, 31, 38, 0.88);
        --card-border: rgba(212, 175, 55, 0.28);
        --text: #f7f7f4;
        --muted: #b9bec7;
        --gold: #d4af37;
    }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(80rem 40rem at 100% -10%, rgba(212,175,55,0.08), transparent 45%),
            radial-gradient(70rem 35rem at -10% 0%, rgba(80,140,255,0.08), transparent 45%),
            linear-gradient(140deg, var(--bg-0) 0%, var(--bg-1) 100%);
        font-family: "Avenir Next", "Manrope", "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(80rem 40rem at 100% -10%, rgba(212,175,55,0.08), transparent 45%),
            radial-gradient(70rem 35rem at -10% 0%, rgba(80,140,255,0.08), transparent 45%),
            linear-gradient(140deg, var(--bg-0) 0%, var(--bg-1) 100%) !important;
    }
    body {
        background: var(--bg-0) !important;
    }
    .block-container { padding-top: 1.6rem !important; }
    h1 {
        margin-bottom: 0.08rem !important;
        letter-spacing: 0.2px;
        font-weight: 760;
    }
    .stCaption {
        margin-top: 0rem !important;
        margin-bottom: 0.22rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.3px;
    }
    .stApp, .stApp * { color: #FFFFFF !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, small, div { color: #FFFFFF !important; }
    .stCaption, .stMarkdown, .stText, .stSubheader { color: #FFFFFF !important; }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stCheckbox"] label { color: #FFFFFF !important; }
    div[data-testid="stVerticalBlock"] > div:has(div.card-style) {
        background: var(--card);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 24px;
        border-radius: 14px;
        border: 1px solid var(--card-border);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        min-height: 0px;
    }
    .card-style {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 0 !important;
        font-size: 0 !important;
    }
    /* Force dark background on inputs */
    div[data-baseweb="select"] > div, div[data-testid="stTextInput"] input {
        background-color: rgba(22, 27, 33, 0.92) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInput"] input {
        color: #111111 !important;
        border-radius: 10px !important;
    }
    /* Dropdown menu options: dark text for readability */
    div[role="listbox"], div[role="listbox"] * {
        color: #111111 !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="popover"] * {
        color: #111111 !important;
    }
    /* Active Button (Gold) vs Inactive (Dark) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f0d46a 0%, #d4af37 60%, #b98f2c 100%) !important;
        color: #111 !important;
        border: 0 !important;
        box-shadow: 0 6px 16px rgba(212,175,55,0.35);
    }
    .stButton > button {
        background-color: rgba(38,39,48,0.92);
        color: #E0E0E0;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(212,175,55,0.55);
        box-shadow: 0 6px 14px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: linear-gradient(180deg, rgba(26,31,38,0.95) 0%, rgba(20,24,30,0.95) 100%);
        border: 1px solid rgba(212,175,55,0.75);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 8px 20px rgba(0,0,0,0.25);
    }
    .gold-text { color: var(--gold); font-weight: 760; }
    .stProgress > div > div > div > div { background-color: #D4AF37 !important; }
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
]
MIN_CAST_ACTORS = 2
LANG_CODES = ["en", "fr", "es", "ja", "de", "other"]
LANG_UI_TO_CODE = {
    "English": "en",
    "Spanish": "es",
    "Mandarin": "other",
    "French": "fr",
    "Korean": "other",
    "Hindi": "other",
}
GENRE_UI_TO_MODEL = {
    "Sci-Fi": "Science Fiction",
    "Action": "Action",
    "Drama": "Drama",
    "Thriller": "Thriller",
    "Comedy": "Comedy",
    "Indie": "Drama",
}


@st.cache_resource
def load_artifacts():
    base = Path(__file__).resolve().parents[1] / "models"
    try:
        with open(base / "popularity_best_model.pkl", "rb") as f:
            pop_model = pickle.load(f)
        with open(base / "ssl_best_model.pkl", "rb") as f:
            ssl_model = pickle.load(f)
        with open(base / "ssl_scaler.pkl", "rb") as f:
            ssl_scaler = pickle.load(f)
        with open(base / "model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    except ModuleNotFoundError as e:
        st.error(
            "Model artifact load failed due to a missing dependency in the deployment environment. "
            f"Missing module: `{e.name}`."
        )
        st.info("Ensure `requirements.txt` includes all model libraries and redeploy the app.")
        st.stop()
    except Exception as e:
        st.error(f"Model artifact load failed: {type(e).__name__}: {e}")
        st.info("Try a clean redeploy and verify model files/version compatibility.")
        st.stop()
    return pop_model, ssl_model, ssl_scaler, metadata


def _tmdb_api_key():
    return TMDB_API_KEY


@st.cache_data(show_spinner=False)
def _tmdb_search_person_id(name: str):
    key = _tmdb_api_key()
    if not key or not name:
        return None
    query = urlencode({"api_key": key, "query": name, "include_adult": "false"})
    url = f"{BASE_URL}/search/person?{query}"
    try:
        with urlopen(url, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return None
    results = payload.get("results", [])
    if not results:
        return None
    return results[0].get("id")


@st.cache_data(show_spinner=False)
def _tmdb_known_for_movies(name: str):
    key = _tmdb_api_key()
    if not key or not name:
        return []
    query = urlencode({"api_key": key, "query": name, "include_adult": "false"})
    url = f"{BASE_URL}/search/person?{query}"
    try:
        with urlopen(url, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return []
    results = payload.get("results", [])
    if not results:
        return []

    known = results[0].get("known_for", [])
    movies = []
    for it in known:
        if it.get("media_type") != "movie":
            continue
        title = it.get("title")
        if not title:
            continue
        popularity = float(it.get("popularity") or 0.0)
        release_date = it.get("release_date") or ""
        movies.append((title, popularity, release_date))

    movies.sort(key=lambda x: (x[1], x[2]), reverse=True)
    out = []
    seen = set()
    for title, _, _ in movies:
        if title not in seen:
            out.append(title)
            seen.add(title)
        if len(out) >= 3:
            break
    return out


@st.cache_data(show_spinner=False)
def _tmdb_top_movies(person_id: int, role: str):
    key = _tmdb_api_key()
    if not key or not person_id:
        return []
    query = urlencode({"api_key": key})
    url = f"{BASE_URL}/person/{person_id}/combined_credits?{query}"
    try:
        with urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return []

    items = payload.get("crew", []) if role == "director" else payload.get("cast", [])
    movies = []
    for it in items:
        if role == "director":
            if it.get("media_type") != "movie":
                continue
            if it.get("job") != "Director":
                continue
        else:
            if it.get("media_type") != "movie":
                continue
        title = it.get("title")
        if not title:
            continue
        popularity = float(it.get("popularity") or 0.0)
        release_date = it.get("release_date") or ""
        movies.append((title, popularity, release_date))

    movies.sort(key=lambda x: (x[1], x[2]), reverse=True)
    uniq = []
    seen = set()
    for title, _, _ in movies:
        if title not in seen:
            uniq.append(title)
            seen.add(title)
        if len(uniq) >= 3:
            break
    return uniq


@st.cache_data(show_spinner=False)
def _tmdb_person_profile(name: str, role: str):
    key = _tmdb_api_key()
    if not key or not name:
        return {}
    query = urlencode({"api_key": key, "query": name, "include_adult": "false"})
    search_url = f"{BASE_URL}/search/person?{query}"
    try:
        with urlopen(search_url, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError):
        return {}
    results = payload.get("results", [])
    if not results:
        return {}

    top = results[0]
    person_id = top.get("id")
    tmdb_popularity = float(top.get("popularity") or 0.0)
    department = top.get("known_for_department") or "N/A"

    movie_credits = None
    if person_id is not None:
        credits_url = f"{BASE_URL}/person/{person_id}/combined_credits?{urlencode({'api_key': key})}"
        try:
            with urlopen(credits_url, timeout=8) as resp:
                credits = json.loads(resp.read().decode("utf-8"))
            if role == "director":
                movie_credits = sum(
                    1
                    for it in credits.get("crew", [])
                    if it.get("media_type") == "movie" and it.get("job") == "Director"
                )
            else:
                movie_credits = sum(1 for it in credits.get("cast", []) if it.get("media_type") == "movie")
        except (URLError, TimeoutError, ValueError):
            movie_credits = None

    return {
        "department": department,
        "tmdb_popularity": tmdb_popularity,
        "movie_credits": movie_credits,
    }


def _get_top_movies(name: str, role: str, fallback_map: dict):
    known_for = _tmdb_known_for_movies(name)
    person_id = _tmdb_search_person_id(name)
    credits = _tmdb_top_movies(person_id, role) if person_id is not None else []

    combined = []
    for t in known_for + credits:
        if t not in combined:
            combined.append(t)
        if len(combined) >= 3:
            break
    if combined:
        return combined[:3]
    return fallback_map.get(name, [])[:3]


def _safe_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


@st.cache_data
def load_people_from_raw():
    raw_path = Path(__file__).resolve().parents[1] / "data" / "movies_2010_2025.csv"
    raw = pd.read_csv(
        raw_path,
        usecols=[
            "title",
            "director_name",
            "director_popularity",
            "director_gender",
            "genres",
            "popularity",
            "runtime",
            "budget",
            "release_date",
            "keywords",
            "overview",
            "actor1_name",
            "actor1_popularity",
            "actor1_gender",
            "actor2_name",
            "actor2_popularity",
            "actor2_gender",
            "actor3_name",
            "actor3_popularity",
            "actor3_gender",
            "actor4_name",
            "actor4_popularity",
            "actor4_gender",
            "actor5_name",
            "actor5_popularity",
            "actor5_gender",
        ],
    )

    raw["genres_parsed"] = raw["genres"].apply(_safe_list)
    raw["release_month"] = pd.to_datetime(raw["release_date"], errors="coerce").dt.month.fillna(7).astype(int)
    raw["keyword_count"] = raw["keywords"].apply(lambda x: len(_safe_list(x)))
    raw["has_overview"] = raw["overview"].fillna("").str.len().gt(0).astype(int)
    raw["overview_length"] = raw["overview"].fillna("").str.len().astype(int)

    director_group = raw.dropna(subset=["director_name"]).groupby("director_name")
    director_data = {}
    director_top_movies = {}
    for name, grp in director_group:
        n = max(len(grp), 1)
        genre_counts = {"Sci-Fi": 0, "Action": 0, "Drama": 0, "Thriller": 0}
        for items in grp["genres_parsed"]:
            s = set(items)
            if "Science Fiction" in s:
                genre_counts["Sci-Fi"] += 1
            if "Action" in s:
                genre_counts["Action"] += 1
            if "Drama" in s:
                genre_counts["Drama"] += 1
            if "Thriller" in s:
                genre_counts["Thriller"] += 1
        genre_scores = {k: int(round(min(10, 10 * (v / n)))) for k, v in genre_counts.items()}
        top_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "N/A"

        dir_pop = float(grp["director_popularity"].median()) if grp["director_popularity"].notna().any() else 1.0
        dir_gender_mode = grp["director_gender"].dropna().mode()
        dir_is_female = int((not dir_gender_mode.empty) and (dir_gender_mode.iloc[0] == 1))
        baseline = float(np.clip(grp["popularity"].median() / 10.0, 1.0, 9.9))

        director_data[name] = {
            "genres": genre_scores,
            "baseline": round(baseline, 1),
            "popularity": dir_pop,
            "is_female": dir_is_female,
            "count": int(n),
            "top_genre": top_genre,
            "runtime_median": float(grp["runtime"].median()) if grp["runtime"].notna().any() else 100.0,
            "budget_median": float(grp["budget"].median()) if grp["budget"].notna().any() else 0.0,
            "release_month_mode": int(grp["release_month"].mode().iloc[0]) if not grp["release_month"].mode().empty else 7,
            "keyword_count_median": int(grp["keyword_count"].median()) if grp["keyword_count"].notna().any() else 5,
            "overview_length_median": int(grp["overview_length"].median()) if grp["overview_length"].notna().any() else 220,
            "has_overview_mode": int(grp["has_overview"].mode().iloc[0]) if not grp["has_overview"].mode().empty else 1,
        }
        top_titles = (
            grp.dropna(subset=["title"])
            .sort_values("popularity", ascending=False)["title"]
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        director_top_movies[name] = top_titles

    actor_rows = []
    for i in range(1, 6):
        sub = raw[[f"actor{i}_name", f"actor{i}_popularity", f"actor{i}_gender"]].copy()
        sub.columns = ["name", "popularity", "gender"]
        actor_rows.append(sub)
    actors = pd.concat(actor_rows, axis=0).dropna(subset=["name"])
    actor_grp = actors.groupby("name")

    actor_data = {}
    actor_top_movies = {}
    for name, grp in actor_grp:
        pop = float(grp["popularity"].median()) if grp["popularity"].notna().any() else 0.0
        g_mode = grp["gender"].dropna().mode()
        is_female = int((not g_mode.empty) and (g_mode.iloc[0] == 1))
        actor_data[name] = {"popularity": pop, "is_female": is_female, "count": int(len(grp))}

    # Actor -> top popular movie titles.
    for i in range(1, 6):
        sub = raw[[f"actor{i}_name", "title", "popularity"]].copy()
        sub.columns = ["name", "title", "popularity"]
        sub = sub.dropna(subset=["name", "title"])
        sub = sub.sort_values("popularity", ascending=False)
        for actor_name, grp in sub.groupby("name"):
            if actor_name not in actor_top_movies:
                actor_top_movies[actor_name] = []
            for t in grp["title"].tolist():
                if t not in actor_top_movies[actor_name]:
                    actor_top_movies[actor_name].append(t)
                if len(actor_top_movies[actor_name]) >= 3:
                    break

    actor_list = sorted(actor_data.keys(), key=lambda n: (-actor_data[n]["count"], n))

    # Fallback if parsing produced empty sets.
    if not director_data:
        director_data = {
            "Christopher Nolan": {"genres": {"Sci-Fi": 10, "Action": 8, "Drama": 7, "Thriller": 9}, "baseline": 5.0, "popularity": 6.5, "is_female": 0, "count": 10, "top_genre": "Sci-Fi"},
            "Greta Gerwig": {"genres": {"Sci-Fi": 4, "Action": 3, "Drama": 10, "Thriller": 5}, "baseline": 4.8, "popularity": 5.2, "is_female": 1, "count": 10, "top_genre": "Drama"},
            "Denis Villeneuve": {"genres": {"Sci-Fi": 10, "Action": 7, "Drama": 8, "Thriller": 9}, "baseline": 4.9, "popularity": 4.8, "is_female": 0, "count": 10, "top_genre": "Sci-Fi"},
        }
        director_top_movies = {
            "Christopher Nolan": ["Inception", "The Dark Knight", "Interstellar"],
            "Greta Gerwig": ["Barbie", "Little Women", "Lady Bird"],
            "Denis Villeneuve": ["Dune", "Blade Runner 2049", "Arrival"],
        }
    if not actor_list:
        actor_data = {
            "Zendaya": {"popularity": 15.0, "is_female": 1, "count": 10},
            "Pedro Pascal": {"popularity": 12.0, "is_female": 0, "count": 10},
            "Timothée Chalamet": {"popularity": 11.0, "is_female": 0, "count": 10},
            "Viola Davis": {"popularity": 8.0, "is_female": 1, "count": 10},
            "Cillian Murphy": {"popularity": 9.0, "is_female": 0, "count": 10},
        }
        actor_list = list(actor_data.keys())
        actor_top_movies = {
            "Zendaya": ["Dune: Part Two", "Spider-Man: No Way Home", "Dune"],
            "Pedro Pascal": ["Gladiator II", "The Wild Robot", "Wonder Woman 1984"],
            "Timothée Chalamet": ["Dune: Part Two", "Wonka", "Dune"],
            "Viola Davis": ["The Woman King", "Suicide Squad", "The Help"],
            "Cillian Murphy": ["Oppenheimer", "Inception", "The Dark Knight"],
        }

    return director_data, actor_data, actor_list, director_top_movies, actor_top_movies


DIRECTOR_DATA, ACTOR_DATA, ACTOR_LIST, DIRECTOR_TOP_MOVIES, ACTOR_TOP_MOVIES = load_people_from_raw()

# --- 2. DATA & SESSION STATE ---
if "director_locked" not in st.session_state:
    st.session_state.director_locked = False
if "selected_dir" not in st.session_state:
    st.session_state.selected_dir = None
if "locked_actors" not in st.session_state:
    st.session_state.locked_actors = []
if "active_genre" not in st.session_state:
    st.session_state.active_genre = "Sci-Fi"
if "active_lang" not in st.session_state:
    st.session_state.active_lang = "English"
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "input_runtime" not in st.session_state:
    st.session_state.input_runtime = 110
if "input_release_year" not in st.session_state:
    st.session_state.input_release_year = 2026
if "input_release_month" not in st.session_state:
    st.session_state.input_release_month = 7
if "input_keyword_count" not in st.session_state:
    st.session_state.input_keyword_count = 5
if "input_budget_musd" not in st.session_state:
    st.session_state.input_budget_musd = 0.0
if "input_overview_length" not in st.session_state:
    st.session_state.input_overview_length = 220


def _clip_runtime(x):
    return int(np.clip(x, 60, 220))


def _build_feature_row(cast_names, overrides=None):
    overrides = overrides or {}
    selected_dir = st.session_state.selected_dir
    d = DIRECTOR_DATA[selected_dir]
    cast_meta = [ACTOR_DATA[a] for a in cast_names if a in ACTOR_DATA]
    cast_pops = [x["popularity"] for x in cast_meta]
    cast_gender = [x["is_female"] for x in cast_meta]

    actor_slots = cast_pops[:5] + [0.0] * max(0, 5 - len(cast_pops))
    runtime = _clip_runtime(overrides.get("runtime", st.session_state.input_runtime))
    release_month = int(np.clip(overrides.get("release_month", st.session_state.input_release_month), 1, 12))
    release_year = int(overrides.get("release_year", st.session_state.input_release_year))
    release_quarter = int((release_month - 1) / 3) + 1
    is_summer_release = int(release_month in [6, 7, 8])
    is_holiday_release = int(release_month in [11, 12])
    keyword_count = int(max(0, overrides.get("keyword_count", st.session_state.input_keyword_count)))
    overview_length = int(max(0, overrides.get("overview_length", st.session_state.input_overview_length)))
    has_overview = int(overrides.get("has_overview", int(overview_length > 0)))
    budget_usd = float(max(0.0, overrides.get("budget_usd", st.session_state.input_budget_musd * 1_000_000.0)))
    has_budget = int(budget_usd > 0)
    budget_missing_flag = int(has_budget == 0)
    log_budget = float(np.log1p(budget_usd)) if has_budget else 0.0
    budget_effective = float(has_budget * log_budget)
    revenue_missing_flag = 1

    model_genre = GENRE_UI_TO_MODEL.get(st.session_state.active_genre, "Drama")
    lang_code = LANG_UI_TO_CODE.get(st.session_state.active_lang, "other")

    row = {
        "runtime": float(runtime),
        "director_popularity": float(d.get("popularity", 1.0)),
        "actor1_popularity": float(actor_slots[0]),
        "actor2_popularity": float(actor_slots[1]),
        "actor3_popularity": float(actor_slots[2]),
        "actor4_popularity": float(actor_slots[3]),
        "actor5_popularity": float(actor_slots[4]),
        "cast_pop_mean": float(np.mean(cast_pops)) if cast_pops else 0.0,
        "cast_pop_max": float(np.max(cast_pops)) if cast_pops else 0.0,
        "release_month": int(release_month),
        "release_year": int(release_year),
        "release_quarter": int(release_quarter),
        "is_summer_release": int(is_summer_release),
        "is_holiday_release": int(is_holiday_release),
        "star_count": int(sum(p >= 5 for p in cast_pops)),
        "cast_popularity_std": float(np.std(cast_pops)) if cast_pops else 0.0,
        "cast_gender_ratio": float(np.mean(cast_gender)) if cast_gender else 0.5,
        "director_is_female": int(d.get("is_female", 0)),
        "num_genres": 1,
        "keyword_count": int(keyword_count),
        "is_english": int(lang_code == "en"),
        "revenue_missing_flag": int(revenue_missing_flag),
        "budget_missing_flag": int(budget_missing_flag),
        "has_budget": int(has_budget),
        "log_budget": float(log_budget),
        "budget_effective": float(budget_effective),
        "has_overview": int(has_overview),
        "overview_length": int(overview_length if has_overview else 0),
    }

    for g in MODEL_GENRES:
        row[f"genre_{g}"] = int(g == model_genre)
    for code in LANG_CODES:
        row[f"lang_{code}"] = int(code == lang_code)
    return row


def _predict_for_cast(cast_names, overrides=None):
    pop_model, ssl_model, ssl_scaler, metadata = load_artifacts()
    row = _build_feature_row(cast_names, overrides=overrides)
    pop_cols = metadata["popularity_feature_cols"]
    ssl_cols = metadata["ssl_feature_cols"]
    pop_transform = metadata.get("popularity_target_transform", "none")

    x_pop = pd.DataFrame([[row.get(c, 0) for c in pop_cols]], columns=pop_cols)
    x_ssl = pd.DataFrame([[row.get(c, 0) for c in ssl_cols]], columns=ssl_cols)
    x_ssl_scaled = pd.DataFrame(ssl_scaler.transform(x_ssl), columns=ssl_cols)

    pop_raw = float(pop_model.predict(x_pop)[0])
    if pop_transform == "log1p":
        popularity_pred = float(np.expm1(pop_raw))
    else:
        popularity_pred = pop_raw
    popularity_pred = float(max(0.0, popularity_pred))
    tier_pred = int(ssl_model.predict(x_ssl_scaled)[0])
    if hasattr(ssl_model, "predict_proba"):
        probas = ssl_model.predict_proba(x_ssl_scaled)[0]
    else:
        probas = np.zeros(4)
        probas[tier_pred] = 1.0

    tier_score_map = {0: 2.5, 1: 4.8, 2: 6.8, 3: 8.8}
    pop_score = float(np.clip(popularity_pred / 10.0, 0.0, 9.9))
    tier_score = tier_score_map.get(tier_pred, 4.5)
    conf = float(np.max(probas)) if len(probas) else 0.0
    combined = float(np.clip(0.65 * pop_score + 0.25 * tier_score + 0.10 * (conf * 10), 0.0, 9.9))

    return {
        "score": round(combined, 1),
        "popularity_pred": popularity_pred,
        "tier_pred": tier_pred,
        "probas": probas,
    }


def _confidence_label(conf):
    if conf >= 0.70:
        return "High"
    if conf >= 0.50:
        return "Medium"
    return "Low"


@st.cache_data
def _popularity_reference():
    p = Path(__file__).resolve().parents[1] / "data" / "data_supervised_popularity.csv"
    vals = pd.read_csv(p, usecols=["popularity"])["popularity"].dropna().astype(float).values
    return np.sort(vals)


@st.cache_data
def _feature_importance_df():
    pop_model, _, _, metadata = load_artifacts()
    feature_cols = metadata["popularity_feature_cols"]
    estimator = pop_model.named_steps.get("model", pop_model)

    if hasattr(estimator, "feature_importances_"):
        vals = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        vals = np.abs(np.ravel(estimator.coef_).astype(float))
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    df = pd.DataFrame({"Feature": feature_cols, "Importance": vals})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


# --- 3. DYNAMIC PREDICTION LOGIC ---
def calculate_hit_prob():
    if not st.session_state.director_locked:
        st.session_state.last_pred = None
        return None
    if len(st.session_state.locked_actors) < MIN_CAST_ACTORS:
        st.session_state.last_pred = None
        return None
    pred = _predict_for_cast(st.session_state.locked_actors)
    st.session_state.last_pred = pred
    return pred["score"]


# --- 4. UI LAYOUT ---
st.title("🎬 Casting Sandbox")
st.caption("PREDICTIVE CAST OPTIMIZER")

st.markdown("## Inputs")
st.markdown('<div class="card-style">', unsafe_allow_html=True)
st.markdown("### Configure Scenario")

row1a, row1b = st.columns([1.2, 0.8], gap="small")
with row1a:
    director_choice = st.selectbox(
        "Director",
        options=sorted(DIRECTOR_DATA.keys()),
        index=(
            sorted(DIRECTOR_DATA.keys()).index(st.session_state.selected_dir)
            if st.session_state.selected_dir in DIRECTOR_DATA
            else 0
        ),
        disabled=st.session_state.director_locked,
    )
with row1b:
    if (not st.session_state.director_locked) and st.button("Lock Director 🔒", use_container_width=True):
        st.session_state.selected_dir = director_choice
        st.session_state.director_locked = True
        d_meta = DIRECTOR_DATA[director_choice]
        st.session_state.input_runtime = int(np.clip(d_meta.get("runtime_median", 110), 60, 220))
        st.session_state.input_release_month = int(np.clip(d_meta.get("release_month_mode", 7), 1, 12))
        st.session_state.input_keyword_count = int(max(0, d_meta.get("keyword_count_median", 5)))
        st.session_state.input_overview_length = int(max(0, d_meta.get("overview_length_median", 220)))
        st.session_state.input_budget_musd = round(float(max(0.0, d_meta.get("budget_median", 0.0))) / 1_000_000.0, 2)
        st.session_state.input_release_year = 2026
        st.rerun()
    if st.session_state.director_locked:
        st.button("Director Locked 🔒", use_container_width=True, disabled=True)

row2a, row2b = st.columns([1.3, 0.7], gap="small")
with row2a:
    selected_cast = st.multiselect(
        "Actors (max 5)",
        options=ACTOR_LIST,
        default=st.session_state.locked_actors,
        max_selections=5,
    )
with row2b:
    st.write("")
    if st.button("Apply Cast", use_container_width=True):
        st.session_state.locked_actors = selected_cast
        st.rerun()
    if st.button("Reset", use_container_width=True):
        st.session_state.director_locked = False
        st.session_state.locked_actors = []
        st.session_state.last_pred = None
        st.rerun()

if st.session_state.director_locked:
    st.write("Top Known-For Movies")
    m1, m2 = st.columns(2, gap="small")
    with m1:
        st.markdown("**Director**")
        d_name = st.session_state.selected_dir
        d_meta = DIRECTOR_DATA.get(d_name, {})
        d_tmdb = _tmdb_person_profile(d_name, "director")
        d_movies = _get_top_movies(d_name, "director", DIRECTOR_TOP_MOVIES)
        d_movies_text = " | ".join(d_movies) if d_movies else "No titles available"
        if d_tmdb:
            d_credits = d_tmdb.get("movie_credits")
            d_credits_txt = str(int(d_credits)) if isinstance(d_credits, (int, float)) else "N/A"
            d_profile = (
                f"TMDB dept: {d_tmdb.get('department', 'N/A')} | "
                f"TMDB popularity: {float(d_tmdb.get('tmdb_popularity', 0.0)):.1f} | "
                f"Movie credits: {d_credits_txt}"
            )
        else:
            d_profile = (
                f"Core genre: {d_meta.get('top_genre', 'N/A')} | "
                f"Movies in dataset: {int(d_meta.get('count', 0))} | "
                f"Median popularity: {float(d_meta.get('popularity', 0.0)):.1f}"
            )
        st.markdown(
            f"<div style='background:#1A1C1E; border:1px solid #444; padding:10px; border-radius:8px;'>"
            f"<strong>{d_name}</strong><br><small>{d_movies_text}</small><br><small>{d_profile}</small></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown("**Cast**")
        if st.session_state.locked_actors:
            for actor in st.session_state.locked_actors:
                a_meta = ACTOR_DATA.get(actor, {})
                a_tmdb = _tmdb_person_profile(actor, "actor")
                a_movies = _get_top_movies(actor, "actor", ACTOR_TOP_MOVIES)
                a_movies_text = " | ".join(a_movies) if a_movies else "No titles available"
                if a_tmdb:
                    a_credits = a_tmdb.get("movie_credits")
                    a_credits_txt = str(int(a_credits)) if isinstance(a_credits, (int, float)) else "N/A"
                    a_profile = (
                        f"TMDB dept: {a_tmdb.get('department', 'N/A')} | "
                        f"TMDB popularity: {float(a_tmdb.get('tmdb_popularity', 0.0)):.1f} | "
                        f"Movie credits: {a_credits_txt}"
                    )
                else:
                    a_profile = (
                        f"Appearances in dataset: {int(a_meta.get('count', 0))} | "
                        f"Median popularity: {float(a_meta.get('popularity', 0.0)):.1f}"
                    )
                st.markdown(
                    f"<div style='background:#1A1C1E; border:1px solid #444; padding:10px; border-radius:8px; margin-bottom:8px;'>"
                    f"<strong>{actor}</strong><br><small>{a_movies_text}</small><br><small>{a_profile}</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Add actors to view known-for movies.")

st.write("Context")
c1, c2 = st.columns(2, gap="small")
with c1:
    st.caption("Genre Focus")
    genres = ["Sci-Fi", "Action", "Drama", "Thriller", "Comedy", "Indie"]
    g_cols = st.columns(3)
    for i, g in enumerate(genres):
        g_type = "primary" if st.session_state.active_genre == g else "secondary"
        if g_cols[i % 3].button(g, key=f"g_{g}", type=g_type, use_container_width=True):
            st.session_state.active_genre = g
            st.rerun()
with c2:
    st.caption("Language")
    langs = ["English", "Spanish", "Mandarin", "French", "Korean", "Hindi"]
    l_cols = st.columns(3)
    for i, l in enumerate(langs):
        l_type = "primary" if st.session_state.active_lang == l else "secondary"
        if l_cols[i % 3].button(l, key=f"l_{l}", type=l_type, use_container_width=True):
            st.session_state.active_lang = l
            st.rerun()

st.write("Advanced Model Inputs")
a1, a2, a3 = st.columns(3)
st.session_state.input_runtime = a1.slider("Runtime", 60, 220, int(st.session_state.input_runtime), key="runtime_slider")
st.session_state.input_release_year = a2.number_input(
    "Release Year", min_value=2010, max_value=2035, value=int(st.session_state.input_release_year), step=1, key="release_year_input"
)
st.session_state.input_release_month = a3.selectbox(
    "Month", options=list(range(1, 13)), index=int(st.session_state.input_release_month) - 1, key="release_month_input"
)
b1, b2, b3 = st.columns(3)
st.session_state.input_keyword_count = b1.slider("Keywords", 0, 60, int(st.session_state.input_keyword_count), key="keyword_slider")
st.session_state.input_budget_musd = b2.number_input(
    "Budget (USD M)", min_value=0.0, max_value=500.0, value=float(st.session_state.input_budget_musd), step=1.0, key="budget_input"
)
st.session_state.input_overview_length = b3.slider(
    "Overview Length", 0, 1000, int(st.session_state.input_overview_length), key="overview_len_slider"
)
st.markdown("</div>", unsafe_allow_html=True)

current_prob = calculate_hit_prob()
if st.session_state.director_locked:
    base_pred = _predict_for_cast([])
    baseline = base_pred["score"]
else:
    base_pred = None
    baseline = 0.0

st.markdown("## Results")
if not st.session_state.director_locked:
    st.info("Lock a director and configure inputs to see predictions.")
elif current_prob is None:
    st.warning(f"Add at least {MIN_CAST_ACTORS} actors to generate predictions.")
else:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    tier_map = {0: "Low", 1: "Medium", 2: "High", 3: "Blockbuster"}
    p = st.session_state.last_pred
    conf = float(np.max(p["probas"])) if len(p["probas"]) else 0.0
    raw_pop = float(p["popularity_pred"])
    pop_ref = _popularity_reference()
    pop_pct = float((pop_ref <= raw_pop).mean() * 100.0) if len(pop_ref) else np.nan
    if len(pop_ref):
        p01 = float(np.percentile(pop_ref, 1))
        p99 = float(np.percentile(pop_ref, 99))
        pop_display = float(np.clip(raw_pop, p01, p99))
    else:
        p01, p99 = 0.0, raw_pop
        pop_display = raw_pop

    pop_margin = max(1.5, 0.08 * abs(pop_display))
    pop_low = max(0.0, float(pop_display) - pop_margin)
    pop_high = float(pop_display) + pop_margin
    diff = round(current_prob - baseline, 1)
    direction = "improvement" if diff > 0 else ("decline" if diff < 0 else "no change")

    st.markdown(
        f"<div style='background:#1A1C1E; border:1px solid #444; padding:12px; border-radius:8px;'>"
        f"<small style='color:gray;'>PREDICTION SUMMARY</small><br>"
        f"Casting Fit Score (0-10): <span class='gold-text'>{current_prob:.1f}</span> "
        f"(<span class='gold-text'>{diff:+.1f}</span> vs director baseline, {direction})<br>"
        f"Predicted popularity: <span class='gold-text'>{pop_display:.1f}</span> "
        f"(higher than <span class='gold-text'>{pop_pct:.1f}%</span> of training movies)<br>"
        f"Revenue outlook: <span class='gold-text'>{tier_map.get(p['tier_pred'], 'N/A')}</span> "
        f"with <span class='gold-text'>{_confidence_label(conf)} confidence ({conf:.0%})</span><br>"
        f"<small style='color:gray;'>Custom model index = 65% popularity + 25% revenue tier + 10% confidence.</small></div>",
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(2, gap="small")

    with chart_left:
        st.write("### Actor and Director Popularity")
        pop_rows = [{"Name": st.session_state.selected_dir, "Role": "Director", "Popularity": float(DIRECTOR_DATA[st.session_state.selected_dir].get("popularity", 0.0))}]
        for actor in st.session_state.locked_actors:
            pop_rows.append(
                {
                    "Name": actor,
                    "Role": "Actor",
                    "Popularity": float(ACTOR_DATA.get(actor, {}).get("popularity", 0.0)),
                }
            )
        pop_chart_df = pd.DataFrame(pop_rows)
        pop_chart = (
            alt.Chart(pop_chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Name:N", sort=None, axis=alt.Axis(labelAngle=-25, title=None)),
                y=alt.Y("Popularity:Q", title="Popularity"),
                color=alt.Color("Role:N", scale=alt.Scale(range=["#d4af37", "#8fa3bf"]), legend=None),
                tooltip=["Name", "Role", alt.Tooltip("Popularity:Q", format=".2f")],
            )
            .properties(height=150)
        )
        st.altair_chart(pop_chart, use_container_width=True)

    with chart_right:
        st.write("### Feature Importance (Global Popularity Model)")
        fi_df = _feature_importance_df().head(12)
        if fi_df.empty:
            st.info("Feature importance not available for the current model type.")
        else:
            fi_chart = (
                alt.Chart(fi_df)
                .mark_bar(color="#d4af37")
                .encode(
                    x=alt.X("Feature:N", sort=None, axis=alt.Axis(labelAngle=-35, labelLimit=120, title=None)),
                    y=alt.Y("Importance:Q", title="Importance"),
                    tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
                )
                .properties(height=170)
            )
            st.altair_chart(fi_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
