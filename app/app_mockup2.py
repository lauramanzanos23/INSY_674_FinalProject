import ast
import json
import os
import pickle
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

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
    .stApp { background-color: #0F1113; color: #FFFFFF; }
    .block-container { padding-top: 0.6rem !important; }
    h1 { margin-bottom: 0.1rem !important; }
    .stCaption { margin-top: 0rem !important; margin-bottom: 0.25rem !important; }
    .stApp, .stApp * { color: #FFFFFF !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, small, div { color: #FFFFFF !important; }
    .stCaption, .stMarkdown, .stText, .stSubheader { color: #FFFFFF !important; }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stCheckbox"] label { color: #FFFFFF !important; }
    div[data-testid="stVerticalBlock"] > div:has(div.card-style) {
        background-color: #1A1C1E; padding: 24px; border-radius: 12px; border: 1px solid #333; min-height: 450px;
    }
    .card-style { display: none !important; }
    /* Force dark background on inputs */
    div[data-baseweb="select"] > div, div[data-testid="stTextInput"] input {
        background-color: #1A1C1E !important; color: #FFFFFF !important; border: 1px solid #444 !important;
    }
    div[data-testid="stNumberInput"] input {
        color: #111111 !important;
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
        background-color: #D4AF37 !important; color: #000 !important; border: none !important;
    }
    .stButton > button {
        background-color: #262730; color: #E0E0E0; border: 1px solid #444; border-radius: 20px;
    }
    .prediction-card {
        background-color: #1A1C1E; border: 1px solid #D4AF37; padding: 20px; border-radius: 8px; text-align: center;
    }
    .gold-text { color: #D4AF37; font-weight: 700; }
    .stProgress > div > div > div > div { background-color: #D4AF37 !important; }
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
    with open(base / "popularity_best_model.pkl", "rb") as f:
        pop_model = pickle.load(f)
    with open(base / "ssl_best_model.pkl", "rb") as f:
        ssl_model = pickle.load(f)
    with open(base / "ssl_scaler.pkl", "rb") as f:
        ssl_scaler = pickle.load(f)
    with open(base / "model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
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

        dir_pop = float(grp["director_popularity"].median()) if grp["director_popularity"].notna().any() else 1.0
        dir_gender_mode = grp["director_gender"].dropna().mode()
        dir_is_female = int((not dir_gender_mode.empty) and (dir_gender_mode.iloc[0] == 1))
        baseline = float(np.clip(grp["popularity"].median() / 10.0, 1.0, 9.9))

        director_data[name] = {
            "genres": genre_scores,
            "baseline": round(baseline, 1),
            "popularity": dir_pop,
            "is_female": dir_is_female,
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
            "Christopher Nolan": {"genres": {"Sci-Fi": 10, "Action": 8, "Drama": 7, "Thriller": 9}, "baseline": 5.0, "popularity": 6.5, "is_female": 0},
            "Greta Gerwig": {"genres": {"Sci-Fi": 4, "Action": 3, "Drama": 10, "Thriller": 5}, "baseline": 4.8, "popularity": 5.2, "is_female": 1},
            "Denis Villeneuve": {"genres": {"Sci-Fi": 10, "Action": 7, "Drama": 8, "Thriller": 9}, "baseline": 4.9, "popularity": 4.8, "is_female": 0},
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

    x_pop = pd.DataFrame([[row.get(c, 0) for c in pop_cols]], columns=pop_cols)
    x_ssl = pd.DataFrame([[row.get(c, 0) for c in ssl_cols]], columns=ssl_cols)
    x_ssl_scaled = pd.DataFrame(ssl_scaler.transform(x_ssl), columns=ssl_cols)

    popularity_pred = float(pop_model.predict(x_pop)[0])
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

col1, col2, col3 = st.columns(3, gap="medium")

# --- TASK 1: THE INTAKE ---
with col1:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    st.markdown("### 👥 The Intake")

    if not st.session_state.director_locked:
        d_choice = st.selectbox(
            "Search Director",
            options=[""] + sorted(DIRECTOR_DATA.keys()),
            label_visibility="collapsed",
        )
        if st.button("Lock Director 🔒", use_container_width=True) and d_choice:
            st.session_state.selected_dir = d_choice
            st.session_state.director_locked = True
            d_meta = DIRECTOR_DATA[d_choice]
            st.session_state.input_runtime = int(np.clip(d_meta.get("runtime_median", 110), 60, 220))
            st.session_state.input_release_month = int(np.clip(d_meta.get("release_month_mode", 7), 1, 12))
            st.session_state.input_keyword_count = int(max(0, d_meta.get("keyword_count_median", 5)))
            st.session_state.input_overview_length = int(max(0, d_meta.get("overview_length_median", 220)))
            st.session_state.input_budget_musd = round(float(max(0.0, d_meta.get("budget_median", 0.0))) / 1_000_000.0, 2)
            st.session_state.input_release_year = 2026
            st.rerun()
    else:
        st.markdown(
            f"""<div style="background:#262730; padding:15px; border-radius:8px; border: 1px solid #D4AF37;">
            <p style="margin:0; font-size:0.7rem; color:gray;">DIRECTOR</p>
            <h4 style="margin:0;">{st.session_state.selected_dir}</h4></div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        scores = DIRECTOR_DATA[st.session_state.selected_dir]["genres"]
        fv1, fv2 = st.columns(2)
        fv1.progress(scores["Sci-Fi"] / 10, text=f"Sci-Fi: {scores['Sci-Fi']}")
        fv2.progress(scores["Action"] / 10, text=f"Action: {scores['Action']}")
        dir_top = _get_top_movies(st.session_state.selected_dir, "director", DIRECTOR_TOP_MOVIES)
        if dir_top:
            st.write("**Top Movies**")
            st.markdown(
                "<div style='font-size:0.9rem; color:#DDD;'>• " + "<br>• ".join(dir_top) + "</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.write("**ADD ACTORS**")
        a_choice = st.selectbox(
            "Search Actor...",
            options=[""] + ACTOR_LIST,
            label_visibility="collapsed",
        )
        if st.button("Add Actor +", use_container_width=True) and a_choice:
            if a_choice not in st.session_state.locked_actors and len(st.session_state.locked_actors) < 5:
                st.session_state.locked_actors.append(a_choice)
                st.rerun()

        if st.button("Reset Selection"):
            st.session_state.director_locked = False
            st.session_state.locked_actors = []
            st.session_state.last_pred = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- TASK 2: CONTEXTUAL SANDBOX (Selectable Buttons) ---
with col2:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Contextual Sandbox")

    st.write("GENRE FOCUS")
    genres = ["Sci-Fi", "Action", "Drama", "Thriller", "Comedy", "Indie"]
    g_cols = st.columns(3)
    for i, g in enumerate(genres):
        g_type = "primary" if st.session_state.active_genre == g else "secondary"
        if g_cols[i % 3].button(g, key=f"g_{g}", type=g_type, use_container_width=True):
            st.session_state.active_genre = g
            st.rerun()

    st.write("LANGUAGES")
    langs = ["English", "Spanish", "Mandarin", "French", "Korean", "Hindi"]
    l_cols = st.columns(3)
    for i, l in enumerate(langs):
        l_type = "primary" if st.session_state.active_lang == l else "secondary"
        if l_cols[i % 3].button(l, key=f"l_{l}", type=l_type, use_container_width=True):
            st.session_state.active_lang = l
            st.rerun()

    st.divider()
    st.write("MODEL INPUTS")
    m1, m2 = st.columns(2)
    st.session_state.input_runtime = m1.slider("Runtime (min)", 60, 220, int(st.session_state.input_runtime), key="runtime_slider")
    st.session_state.input_release_year = m2.number_input(
        "Release Year",
        min_value=2010,
        max_value=2035,
        value=int(st.session_state.input_release_year),
        step=1,
        key="release_year_input",
    )

    m3, m4 = st.columns(2)
    st.session_state.input_release_month = m3.selectbox(
        "Release Month",
        options=list(range(1, 13)),
        index=int(st.session_state.input_release_month) - 1,
        key="release_month_input",
    )
    m4.caption("Num Genres: inferred from selected context")

    m5, m6 = st.columns(2)
    st.session_state.input_keyword_count = m5.slider("Keyword Count", 0, 60, int(st.session_state.input_keyword_count), key="keyword_slider")
    st.session_state.input_budget_musd = m6.number_input(
        "Budget (USD M)",
        min_value=0.0,
        max_value=500.0,
        value=float(st.session_state.input_budget_musd),
        step=1.0,
        key="budget_input",
    )

    m7, _ = st.columns(2)
    st.session_state.input_overview_length = m7.slider(
        "Overview Length",
        0,
        1000,
        int(st.session_state.input_overview_length),
        key="overview_len_slider",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- TASK 3: RANK OPTIMIZER ---
with col3:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    st.markdown("### 📊 Rank Optimizer")
    if not st.session_state.director_locked:
        st.markdown(
            "<div style='text-align:center; padding-top:100px; color:#555;'>Lock a director first...</div>",
            unsafe_allow_html=True,
        )
    else:
        for i, actor in enumerate(st.session_state.locked_actors):
            actor_top = _get_top_movies(actor, "actor", ACTOR_TOP_MOVIES)
            movies_html = ""
            if actor_top:
                movies_html = "<br><small style='color:#BBB;'>Top: " + " | ".join(actor_top) + "</small>"
            st.markdown(
                f"<div style='background:#1A1C1E; border: 1px solid #444; padding:10px; border-radius:8px; margin-bottom:8px;'>#{i+1} | 👤 {actor}{movies_html}</div>",
                unsafe_allow_html=True,
            )
        if len(st.session_state.locked_actors) < MIN_CAST_ACTORS:
            st.markdown(
                f"<div style='background:#1A1C1E; border:1px solid #444; padding:10px; border-radius:8px; margin-top:12px; color:#FFC107;'>"
                f"Add at least {MIN_CAST_ACTORS} actors to score Current Config."
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.session_state.last_pred is not None:
            tier_map = {0: "Low", 1: "Medium", 2: "High", 3: "Blockbuster"}
            p = st.session_state.last_pred
            conf = float(np.max(p["probas"])) if len(p["probas"]) else 0.0
            pop_margin = max(1.5, 0.08 * float(p["popularity_pred"]))
            pop_low = max(0.0, float(p["popularity_pred"]) - pop_margin)
            pop_high = float(p["popularity_pred"]) + pop_margin
            st.markdown(
                f"<div style='background:#1A1C1E; border: 1px solid #444; padding:10px; border-radius:8px; margin-top:12px;'>"
                f"<small style='color:gray;'>MODEL OUTPUT</small><br>"
                f"Popularity: <span class='gold-text'>{p['popularity_pred']:.1f}</span><br>"
                f"Revenue Tier: <span class='gold-text'>{tier_map.get(p['tier_pred'], 'N/A')}</span><br>"
                f"Confidence: <span class='gold-text'>{_confidence_label(conf)} ({conf:.0%})</span><br>"
                f"Popularity range: <span class='gold-text'>{pop_low:.1f} - {pop_high:.1f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            base_pred_viz = _predict_for_cast([])
            tier_names = ["Low", "Medium", "High", "Blockbuster"]
            prob_vals = [float(p["probas"][i]) if i < len(p["probas"]) else 0.0 for i in range(4)]
            prob_pct = [round(v * 100, 1) for v in prob_vals]
            prob_df = pd.DataFrame({"Tier": tier_names, "Probability %": prob_pct})

            st.write("**Revenue Tier Probabilities**")
            for i, tier in enumerate(tier_names):
                st.progress(min(max(prob_vals[i], 0.0), 1.0), text=f"{tier}: {prob_pct[i]:.1f}%")

            if st.session_state.locked_actors:
                actor_df = pd.DataFrame(
                    {
                        "Actor": st.session_state.locked_actors,
                        "TMDB Popularity": [round(float(ACTOR_DATA.get(a, {}).get("popularity", 0.0)), 2) for a in st.session_state.locked_actors],
                    }
                ).set_index("Actor")
                st.write("**Selected Cast Popularity**")
                st.bar_chart(actor_df)

            baseline_blockbuster = float(base_pred_viz["probas"][3]) if len(base_pred_viz["probas"]) > 3 else 0.0
            current_blockbuster = float(p["probas"][3]) if len(p["probas"]) > 3 else 0.0
            compare_df = pd.DataFrame(
                {
                    "Metric": ["Score", "Popularity Pred", "Blockbuster Prob %"],
                    "Baseline": [
                        round(base_pred_viz["score"], 2),
                        round(base_pred_viz["popularity_pred"], 2),
                        round(baseline_blockbuster * 100, 1),
                    ],
                    "Current": [
                        round(p["score"], 2),
                        round(p["popularity_pred"], 2),
                        round(current_blockbuster * 100, 1),
                    ],
                }
            )
            compare_df["Delta"] = (compare_df["Current"] - compare_df["Baseline"]).round(2)
            st.write("**Baseline vs Current (What Changed)**")
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            base_row = _build_feature_row([])
            cur_row = _build_feature_row(st.session_state.locked_actors)
            driver_rows = [
                ("Cast Avg Popularity", cur_row["cast_pop_mean"] - base_row["cast_pop_mean"]),
                ("Star Count", cur_row["star_count"] - base_row["star_count"]),
                ("Lead Actor Popularity", cur_row["actor1_popularity"] - base_row["actor1_popularity"]),
                ("Budget (log)", cur_row["log_budget"] - base_row["log_budget"]),
                ("Release Year vs 2020", cur_row["release_year"] - 2020),
                ("Runtime vs 110", cur_row["runtime"] - 110),
            ]
            driver_df = pd.DataFrame(driver_rows, columns=["Driver", "Signal"])
            driver_df["AbsSignal"] = driver_df["Signal"].abs()
            top_driver_df = driver_df.sort_values("AbsSignal", ascending=False).head(3).copy()
            top_driver_df["Direction"] = top_driver_df["Signal"].apply(lambda x: "Up" if x >= 0 else "Down")
            st.write("**Top 3 Driver Signals**")
            st.dataframe(top_driver_df[["Driver", "Signal", "Direction"]], use_container_width=True, hide_index=True)

            month_rows = []
            for m in range(1, 13):
                m_pred = _predict_for_cast(st.session_state.locked_actors, overrides={"release_month": m})
                month_rows.append({"ReleaseMonth": m, "Score": m_pred["score"]})
            month_df = pd.DataFrame(month_rows).set_index("ReleaseMonth")
            st.write("**Sensitivity: Score vs Release Month**")
            st.line_chart(month_df)
    st.markdown("</div>", unsafe_allow_html=True)

# --- PREDICTION ENGINE (Bottom) ---
st.divider()
bot1, bot2, bot3 = st.columns([1, 1, 2])
current_prob = calculate_hit_prob()

with bot1:
    if st.session_state.director_locked:
        base_pred = _predict_for_cast([])
        baseline = base_pred["score"]
    else:
        baseline = 0.0
    st.markdown(
        f'<div style="background:#1A1C1E; padding:20px; border-radius:8px; border:1px solid #333;"><small style="color:gray;">DIRECTOR BASELINE</small><h2 style="margin:0;">{baseline}</h2></div>',
        unsafe_allow_html=True,
    )

with bot2:
    st.markdown(
        f'<div class="prediction-card"><small class="gold-text">CURRENT CONFIG</small><h2 style="margin:0;" class="gold-text">{current_prob if current_prob is not None else "—"}</h2></div>',
        unsafe_allow_html=True,
    )

with bot3:
    if st.session_state.director_locked and current_prob is not None:
        diff = round(current_prob - baseline, 1)
        pred = st.session_state.last_pred or {}
        probs = pred.get("probas", np.zeros(4))
        tier_names = ["Low", "Medium", "High", "Blockbuster"]
        prob_text = " | ".join(
            [f"{tier_names[i]} {float(probs[i])*100:.0f}%" if i < len(probs) else f"{tier_names[i]} 0%" for i in range(4)]
        )
        st.markdown(
            f'<div style="background:#1A1C1E; padding:15px; border-radius:8px; border:1px solid #333; color:#4CAF50;">📈 +{diff} <span style="color:gray;">vs baseline</span><br><span style="color:#AAA;">{prob_text}</span></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.director_locked:
        st.markdown(
            f'<div style="background:#1A1C1E; padding:15px; border-radius:8px; border:1px solid #333; color:#FFC107;">'
            f'Add at least {MIN_CAST_ACTORS} actors to compare vs baseline.'
            f"</div>",
            unsafe_allow_html=True,
        )
