#APP.PY
import streamlit as st

# -------------------------
# Page config + minimal styling
# -------------------------
st.set_page_config(page_title="The Next Blockbuster", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .card {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
      }
      .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(120,30,60,0.35), rgba(20,20,40,0.35));
        border: 1px solid rgba(255,255,255,0.10);
      }
      .subtle {opacity: 0.85;}
      .label {
        display:inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        margin-right: 0.35rem;
      }
      .big {font-size: 2.2rem; font-weight: 800; line-height: 1;}
      .small {font-size: 0.95rem; opacity: 0.9;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Mock predictor (no ML deps)
# -------------------------
def mock_predict(genre: str, actor_tier: str, runtime: int, release_month: int):
    """
    Returns:
      hit_prob: 0..1
      expected_popularity: 0..100 (mock)
      explanation: list[str]
    """
    score = 0.35  # base
    explanation = []

    # Actor tier signal
    tier_points = {"Unknown": 0.00, "Rising": 0.08, "Star": 0.18, "Superstar": 0.28}
    score += tier_points.get(actor_tier, 0.0)
    explanation.append(f"Actor tier: {actor_tier} ({tier_points.get(actor_tier, 0.0):+.2f})")

    # Genre signal (purely illustrative)
    genre_boost = {
        "Action": 0.10, "Adventure": 0.08, "Animation": 0.09, "Comedy": 0.06,
        "Drama": 0.03, "Horror": 0.07, "Thriller": 0.06, "Romance": 0.02,
        "Science Fiction": 0.08, "Fantasy": 0.06, "Documentary": -0.02,
        "Crime": 0.04, "Family": 0.04, "Mystery": 0.04, "History": 0.01,
        "War": 0.00, "Western": -0.01, "Music": 0.00, "TV Movie": -0.03
    }
    gb = genre_boost.get(genre, 0.0)
    score += gb
    explanation.append(f"Genre: {genre} ({gb:+.2f})")

    # Runtime sweet spot
    if 95 <= runtime <= 135:
        score += 0.06
        explanation.append("Runtime in sweet spot 95–135 min (+0.06)")
    elif runtime > 160:
        score -= 0.04
        explanation.append("Very long runtime >160 min (-0.04)")
    else:
        explanation.append("Runtime neutral (+0.00)")

    # Seasonality (illustrative)
    if release_month in [6, 7, 11, 12]:
        score += 0.05
        explanation.append("Release timing: peak season (+0.05)")
    else:
        explanation.append("Release timing: neutral (+0.00)")

    # Clamp
    hit_prob = max(0.05, min(0.95, score))

    # Convert to an easy-to-read "expected popularity" mock scale
    expected_popularity = 10 + hit_prob * 90

    return hit_prob, expected_popularity, explanation

def label_from_prob(p: float) -> str:
    if p >= 0.75:
        return "High Potential"
    if p >= 0.55:
        return "Promising"
    if p >= 0.40:
        return "Moderate"
    return "High Risk"

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <div class="hero">
      <div class="big">🎬 The Next Blockbuster</div>
      <div class="small subtle">
        Mock demo UI for a pre-release success predictor (genre + actor). Replace the scoring logic with your trained model later.
      </div>
      <div style="margin-top:0.6rem;">
        <span class="label">Pre-release</span>
        <span class="label">Business-oriented</span>
        <span class="label">End-to-end pipeline</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input: movie concept")

    genre = st.selectbox(
        "Primary genre",
        [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
            "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
            "Science Fiction", "TV Movie", "Thriller", "War", "Western"
        ],
        index=0,
        help="In the real system this becomes multi-hot genre features."
    )

    actor_tier = st.radio(
        "Lead actor market tier (mock input)",
        ["Unknown", "Rising", "Star", "Superstar"],
        index=2,
        horizontal=True,
        help="Mock input for actor strength. In the real system this comes from TMDB actor popularity."
    )

    runtime = st.slider("Runtime (minutes)", 60, 220, 115, step=5)
    release_month = st.select_slider("Release month", options=list(range(1, 13)), value=6)

    st.caption("Note: This UI is dependency-free. Replace `mock_predict()` with your trained model later.")
    run = st.button("Predict success", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output: decision support")

    if run:
        p_hit, exp_pop, why = mock_predict(genre, actor_tier, runtime, release_month)
        verdict = label_from_prob(p_hit)

        st.metric("Hit probability", f"{p_hit:.0%}")
        st.metric("Expected popularity (mock)", f"{exp_pop:.1f} / 100")
        st.success(f"Recommendation: **{verdict}**")

        st.write("**Drivers (explainable summary):**")
        for w in why:
            st.write(f"- {w}")

        st.divider()
        st.write("**Suggested business action (example):**")
        if p_hit >= 0.75:
            st.write("Prioritize marketing and distribution planning early; consider premium release windows.")
        elif p_hit >= 0.55:
            st.write("Proceed with standard investment; run sensitivity checks on casting and release timing.")
        else:
            st.write("Treat as higher risk; explore alternative casting/genre positioning or reduce spend.")
    else:
        st.info("Select inputs on the left and click **Predict success**.")
        st.caption("This is a mock demo. No API calls and no ML models are used yet.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption("© Demo UI for course project proposal — replace mock scoring with your trained regression/classification models.")
