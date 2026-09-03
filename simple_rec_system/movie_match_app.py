import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Movie Match", page_icon="🍿", layout="centered")

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .crazy-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6ec4, #7873f5, #4facfe, #ff6ec4);
        background-size: 300% 300%;
        animation: gradient-shift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #999;
        margin-top: 0;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
    }
    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        font-size: 1.2rem;
        font-weight: 800;
        border-radius: 999px;
        background: linear-gradient(90deg, #ff6ec4, #7873f5);
        color: white;
        border: none;
        transition: transform 0.15s ease, box-shadow 0.2s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(120, 115, 245, 0.5);
    }
    .movie-card {
        background: linear-gradient(135deg, #fdf0ff, #eef1ff);
        border: 2px solid #e0d4ff;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 10px;
        transition: transform 0.15s ease;
    }
    .movie-card:hover {
        transform: scale(1.04) rotate(-1deg);
    }
    .no-rec-card {
        background: #fff3f3;
        border: 2px dashed #ffb3b3;
        border-radius: 18px;
        padding: 24px;
        text-align: center;
        font-size: 1.2rem;
        color: #b23b3b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='crazy-title'>🍿 MOVIE MATCH 🎬</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Find your movie twin with cosine similarity, then steal their recommendations</p>", unsafe_allow_html=True)

MOVIE_EMOJI = ["🌀", "🦇", "🚀", "💑", "🍻", "🎥", "🍿", "🎞️", "🕶️", "👽"]

# ---------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------
default_users = ["Nikita", "Javid", "Yuvraj", "Kirti", "Sanik"]
default_movies = ["Inception", "The Dark Knight", "Interstellar", "The Notebook", "The Hangover"]
default_ratings = pd.DataFrame(
    [
        [0, 1, 2, 3, 5],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 0],
        [3, 4, 5, 0, 1],
        [4, 5, 0, 1, 2],
    ],
    index=default_users,
    columns=default_movies,
)

with st.expander("🎛️ Ratings matrix (0 = unwatched, 1-5 = rating)"):
    edited_df = st.data_editor(default_ratings, num_rows="dynamic", use_container_width=True)
    st.caption("Rows = users, columns = movies. Edit freely, or add new rows/columns.")

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

ratings = edited_df.to_numpy(dtype=float)
users = np.array(edited_df.index.astype(str))
movies = np.array(edited_df.columns.astype(str))

if len(users) < 2:
    st.warning("Add at least 2 users to find a match.")
    st.stop()

st.write(f"**Shape:** {ratings.shape} · **Dimensions:** {ratings.ndim} · **Total values:** {ratings.size}")

avg_rating = np.mean(ratings, axis=1)
st.dataframe(
    pd.DataFrame({"User": users, "Avg Rating": avg_rating}).set_index("User").round(2),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Pick target user
# ---------------------------------------------------------------------
st.subheader("🎯 Who are we finding a match for?")
target_name = st.selectbox("Target user", users, index=0)
target_idx = int(np.where(users == target_name)[0][0])
target_rating = ratings[target_idx]

find_clicked = st.button("✨ FIND MY MOVIE TWIN ✨")

# ---------------------------------------------------------------------
# Similarity + recommendations
# ---------------------------------------------------------------------
if find_clicked or "twin_result" in st.session_state:
    with st.spinner("Crunching cosine similarities..."):
        similarity_record = []
        for i in range(len(users)):
            vector_rating = ratings[i]
            dot_product = np.dot(target_rating, vector_rating)
            mag1 = np.linalg.norm(target_rating)
            mag2 = np.linalg.norm(vector_rating)
            similarity = dot_product / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0
            similarity_record.append(similarity)

        similarity_record = np.array(similarity_record)
        similarity_record_for_ranking = similarity_record.copy()
        similarity_record_for_ranking[target_idx] = -1.0  # exclude self
        most_similar_idx = int(np.argmax(similarity_record_for_ranking))

    st.session_state.twin_result = True

    # -------------------------------------------------------------
    # Similarity bar chart
    # -------------------------------------------------------------
    st.subheader("💫 Similarity to " + target_name)
    colors = ["#7873f5" if i != target_idx else "#dddddd" for i in range(len(users))]
    colors[most_similar_idx] = "#ff6ec4"

    fig_sim = go.Figure(go.Bar(x=users, y=similarity_record, marker_color=colors))
    fig_sim.update_layout(yaxis_title="Cosine similarity", yaxis_range=[-1, 1])
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown(
        f"<h3 style='text-align:center;'>🎉 Your movie twin is <span style='color:#ff6ec4;'>{users[most_similar_idx]}</span>! 🎉</h3>",
        unsafe_allow_html=True,
    )
    st.balloons()

    # -------------------------------------------------------------
    # Radar chart: taste profile overlay
    # -------------------------------------------------------------
    st.subheader("🕸️ Taste Profile Overlay")
    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(r=target_rating, theta=movies, fill="toself", name=target_name, line_color="#7873f5")
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=ratings[most_similar_idx], theta=movies, fill="toself", name=users[most_similar_idx], line_color="#ff6ec4"
        )
    )
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # -------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------
    st.subheader(f"🍿 Recommended for {target_name}")

    unwatched_mask = target_rating == 0
    similar_user_rating = ratings[most_similar_idx]
    liked_mask = similar_user_rating >= 4
    rec_mask = liked_mask & unwatched_mask
    recommended_movies = movies[rec_mask]

    if len(recommended_movies) == 0:
        st.markdown(
            "<div class='no-rec-card'>🤷 No unwatched movies that your twin loved (rated 4+) — "
            "try a different target user or add more movies!</div>",
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(min(3, len(recommended_movies)))
        for i, movie in enumerate(recommended_movies):
            emoji = MOVIE_EMOJI[i % len(MOVIE_EMOJI)]
            with cols[i % len(cols)]:
                st.markdown(f"<div class='movie-card'>{emoji}<br>{movie}</div>", unsafe_allow_html=True)
        st.snow()

    with st.expander("🔍 See the raw masks"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Movie": movies,
                    f"{target_name} unwatched": unwatched_mask,
                    f"{users[most_similar_idx]} liked (≥4)": liked_mask,
                    "Recommended": rec_mask,
                }
            ).set_index("Movie"),
            use_container_width=True,
        )
