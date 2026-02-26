import streamlit as st

# --- 1. PAGE SETUP & DESIGN ---
st.set_page_config(page_title="Casting Sandbox", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0F1113; color: #FFFFFF; }
    div[data-testid="stVerticalBlock"] > div:has(div.card-style) {
        background-color: #1A1C1E; padding: 24px; border-radius: 12px; border: 1px solid #333; min-height: 450px;
    }
    /* Force dark background on inputs */
    div[data-baseweb="select"] > div, div[data-testid="stTextInput"] input {
        background-color: #1A1C1E !important; color: #FFFFFF !important; border: 1px solid #444 !important;
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
    """, unsafe_allow_html=True)

# --- 2. DATA & SESSION STATE ---
DIRECTOR_DATA = {
    "Christopher Nolan": {"genres": {"Sci-Fi": 10, "Action": 8, "Drama": 7, "Thriller": 9}, "baseline": 5.0},
    "Greta Gerwig": {"genres": {"Sci-Fi": 4, "Action": 3, "Drama": 10, "Thriller": 5}, "baseline": 4.8},
    "Denis Villeneuve": {"genres": {"Sci-Fi": 10, "Action": 7, "Drama": 8, "Thriller": 9}, "baseline": 4.9}
}
ACTOR_LIST = ["Zendaya", "Pedro Pascal", "Timothée Chalamet", "Viola Davis", "Cillian Murphy"]

# Initialize States
if 'director_locked' not in st.session_state: st.session_state.director_locked = False
if 'selected_dir' not in st.session_state: st.session_state.selected_dir = None
if 'locked_actors' not in st.session_state: st.session_state.locked_actors = []
if 'active_genre' not in st.session_state: st.session_state.active_genre = "Sci-Fi"
if 'active_lang' not in st.session_state: st.session_state.active_lang = "English"

# --- 3. DYNAMIC PREDICTION LOGIC ---
def calculate_hit_prob():
    if not st.session_state.director_locked: return 0.0
    
    # Base from Director
    score = DIRECTOR_DATA[st.session_state.selected_dir]["baseline"]
    
    # Impact of Actors (Weighted by count for POC)
    score += (len(st.session_state.locked_actors) * 0.3)
    
    # Impact of Contextual Sandbox (Genre/Language Match)
    if st.session_state.active_genre in ["Sci-Fi", "Action"]: score += 0.4
    if st.session_state.active_lang != "English": score += 0.2
    
    return round(min(score, 9.9), 1)

# --- 4. UI LAYOUT ---
st.title("🎬 Casting Sandbox")
st.caption("PREDICTIVE CAST OPTIMIZER")

col1, col2, col3 = st.columns(3, gap="medium")

# --- TASK 1: THE INTAKE ---
with col1:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    st.markdown("### 👥 The Intake")
    
    if not st.session_state.director_locked:
        d_choice = st.selectbox("Search Director", options=[""] + list(DIRECTOR_DATA.keys()), label_visibility="collapsed")
        if st.button("Lock Director 🔒", use_container_width=True) and d_choice:
            st.session_state.selected_dir = d_choice
            st.session_state.director_locked = True
            st.rerun()
    else:
        st.markdown(f"""<div style="background:#262730; padding:15px; border-radius:8px; border: 1px solid #D4AF37;">
            <p style="margin:0; font-size:0.7rem; color:gray;">DIRECTOR</p>
            <h4 style="margin:0;">{st.session_state.selected_dir}</h4></div>""", unsafe_allow_html=True)
        
        # Feature Vector
        st.write("")
        scores = DIRECTOR_DATA[st.session_state.selected_dir]["genres"]
        fv1, fv2 = st.columns(2)
        fv1.progress(scores["Sci-Fi"]/10, text=f"Sci-Fi: {scores['Sci-Fi']}")
        fv2.progress(scores["Action"]/10, text=f"Action: {scores['Action']}")
        
        st.divider()
        st.write("**ADD ACTORS**")
        a_choice = st.selectbox("Search Actor...", options=[""] + ACTOR_LIST, label_visibility="collapsed")
        if st.button("Add Actor +", use_container_width=True) and a_choice:
            if a_choice not in st.session_state.locked_actors:
                st.session_state.locked_actors.append(a_choice)
                st.rerun()
        
        if st.button("Reset Selection"):
            st.session_state.director_locked = False
            st.session_state.locked_actors = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- TASK 2: CONTEXTUAL SANDBOX (Selectable Buttons) ---
with col2:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Contextual Sandbox")
    
    st.write("GENRE FOCUS")
    genres = ["Sci-Fi", "Action", "Drama", "Thriller", "Comedy", "Indie"]
    g_cols = st.columns(3)
    for i, g in enumerate(genres):
        # Set type to 'primary' if this genre is active
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
    st.markdown('</div>', unsafe_allow_html=True)

# --- TASK 3: RANK OPTIMIZER ---
with col3:
    st.markdown('<div class="card-style">', unsafe_allow_html=True)
    st.markdown("### 📊 Rank Optimizer")
    if not st.session_state.director_locked:
        st.markdown("<div style='text-align:center; padding-top:100px; color:#555;'>Lock a director first...</div>", unsafe_allow_html=True)
    else:
        for i, actor in enumerate(st.session_state.locked_actors):
            st.markdown(f"<div style='background:#1A1C1E; border: 1px solid #444; padding:10px; border-radius:8px; margin-bottom:8px;'>#{i+1} | 👤 {actor}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION ENGINE (Bottom) ---
st.divider()
bot1, bot2, bot3 = st.columns([1, 1, 2])
current_prob = calculate_hit_prob()

with bot1:
    baseline = DIRECTOR_DATA[st.session_state.selected_dir]["baseline"] if st.session_state.director_locked else 0.0
    st.markdown(f'<div style="background:#1A1C1E; padding:20px; border-radius:8px; border:1px solid #333;"><small style="color:gray;">DIRECTOR BASELINE</small><h2 style="margin:0;">{baseline}</h2></div>', unsafe_allow_html=True)

with bot2:
    st.markdown(f'<div class="prediction-card"><small class="gold-text">CURRENT CONFIG</small><h2 style="margin:0;" class="gold-text">{current_prob if st.session_state.director_locked else "—"}</h2></div>', unsafe_allow_html=True)

with bot3:
    if st.session_state.director_locked:
        diff = round(current_prob - baseline, 1)
        st.markdown(f'<div style="background:#1A1C1E; padding:15px; border-radius:8px; border:1px solid #333; color:#4CAF50;">📈 +{diff} <span style="color:gray;">vs baseline</span></div>', unsafe_allow_html=True)