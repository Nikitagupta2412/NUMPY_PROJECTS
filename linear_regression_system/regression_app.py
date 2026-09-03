import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Study Hours → Score Predictor", page_icon="🚀", layout="centered")

# ---------------------------------------------------------------------
# CRAZY styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes wiggle {
        0%, 100% { transform: rotate(-1deg); }
        50% { transform: rotate(1deg); }
    }
    @keyframes pop-in {
        0% { transform: scale(0.5); opacity: 0; }
        70% { transform: scale(1.08); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    .crazy-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6ec4, #7873f5, #4facfe, #43e97b, #ff6ec4);
        background-size: 400% 400%;
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
        font-size: 1.15rem;
        font-weight: 600;
    }
    div.stButton > button {
        width: 100%;
        height: 3.4rem;
        font-size: 1.25rem;
        font-weight: 800;
        border-radius: 999px;
        background: linear-gradient(90deg, #ff6ec4, #7873f5, #4facfe);
        background-size: 200% 200%;
        animation: gradient-shift 3s ease infinite;
        color: white;
        border: none;
        transition: transform 0.15s ease, box-shadow 0.2s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 24px rgba(120, 115, 245, 0.55);
    }
    .stat-card {
        background: linear-gradient(135deg, #fdf0ff, #eef1ff);
        border: 2px solid #e0d4ff;
        border-radius: 18px;
        padding: 16px;
        text-align: center;
        font-weight: 700;
        color: #1b2a6b;
        animation: pop-in 0.4s ease;
        transition: transform 0.15s ease;
    }
    .stat-card:hover {
        transform: scale(1.04) rotate(-1deg);
    }
    .good-card {
        background: linear-gradient(135deg, #e2f9ee, #d4f5e3);
        border: 2px solid #7be0a8;
        border-radius: 18px;
        padding: 18px;
        color: #1b2a6b;
        animation: pop-in 0.4s ease;
    }
    .bad-card {
        background: linear-gradient(135deg, #fff0f0, #ffe0e0);
        border: 2px solid #ff9d9d;
        border-radius: 18px;
        padding: 18px;
        color: #1b2a6b;
        animation: pop-in 0.4s ease;
    }
    .verdict-banner {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 900;
        padding: 14px;
        border-radius: 16px;
        animation: pop-in 0.5s ease, wiggle 2.5s ease-in-out infinite;
        margin: 10px 0 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='crazy-title'>🚀 STUDY-O-METER 3000 🚀</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Turn study hours into predicted scores with pure linear regression magic ✨</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Input: editable training data
# ---------------------------------------------------------------------
default_data = pd.DataFrame(
    {"Study Hours": [2, 3, 5, 7, 8], "Actual Score": [50, 65, 75, 88, 95]}
)

with st.expander("📊 Training Data (tap to edit)", expanded=False):
    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, hide_index=True)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every row has both values filled in.")
    st.stop()

if len(edited_df) < 2:
    st.warning("Add at least 2 data points to fit a line.")
    st.stop()

study_hours = edited_df["Study Hours"].to_numpy(dtype=float)
actual_scores = edited_df["Actual Score"].to_numpy(dtype=float)

crunch_clicked = st.button("🔥 CRUNCH THE NUMBERS 🔥")

if crunch_clicked or "crunched" in st.session_state:
    st.session_state.crunched = True

    with st.spinner("Summoning the regression line..."):
        stuhours_mean = np.mean(study_hours)
        actscores_mean = np.mean(actual_scores)

        denom = np.sum((study_hours - stuhours_mean) ** 2)
        if denom == 0:
            st.error("All Study Hours values are identical, so a slope can't be computed (division by zero).")
            st.stop()

        slope = np.sum((study_hours - stuhours_mean) * (actual_scores - actscores_mean)) / denom
        intercept = actscores_mean - slope * stuhours_mean

        predicted_score = (slope * study_hours) + intercept
        errors = predicted_score - actual_scores
        abs_error = np.abs(errors)
        mae = np.mean(abs_error)
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)

    # -------------------------------------------------------------
    # Fitted line equation
    # -------------------------------------------------------------
    st.subheader("⚡ The Fitted Line")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='stat-card'>SLOPE 📈<br><span style='font-size:1.8rem;'>{slope:.3f}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'>INTERCEPT 🎯<br><span style='font-size:1.8rem;'>{intercept:.3f}</span></div>", unsafe_allow_html=True)
    st.latex(f"\\text{{score}} = {slope:.3f} \\times \\text{{hours}} + {intercept:.3f}")

    # -------------------------------------------------------------
    # Scatter + fitted line
    # -------------------------------------------------------------
    line_x = np.linspace(min(study_hours.min(), 0), study_hours.max() + 1, 100)
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Fitted line", line=dict(color="#7873f5", width=4)))
    fig.add_trace(
        go.Scatter(
            x=study_hours,
            y=actual_scores,
            mode="markers",
            name="Actual scores",
            marker=dict(size=16, color="#ff6ec4", line=dict(width=2, color="white")),
        )
    )
    fig.update_layout(title="📊 Study Hours vs Score", xaxis_title="Study Hours", yaxis_title="Score", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # Vibe check gauge
    # -------------------------------------------------------------
    st.subheader("🌡️ Model Vibe Check")
    max_possible_error = max(np.max(np.abs(actual_scores - actscores_mean)), 1)
    accuracy_score = float(np.clip(100 - (rmse / max_possible_error) * 100, 0, 100))

    gauge_color = "#43e97b" if accuracy_score >= 80 else ("#f2a65a" if accuracy_score >= 60 else "#e57373")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=accuracy_score,
            number={"suffix": " / 100", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 60], "color": "#fbe4e4"},
                    {"range": [60, 80], "color": "#fdf0e0"},
                    {"range": [80, 100], "color": "#e2f9ee"},
                ],
            },
        )
    )
    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    if accuracy_score >= 80:
        st.markdown("<div class='verdict-banner' style='background:#e2f9ee; color:#1a7a4c;'>🔥 THIS MODEL IS ON FIRE! 🔥</div>", unsafe_allow_html=True)
        st.balloons()
    elif accuracy_score >= 60:
        st.markdown("<div class='verdict-banner' style='background:#fdf0e0; color:#a15d1a;'>😎 Pretty solid, not gonna lie.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-banner' style='background:#fbe4e4; color:#a12a2a;'>😬 This model needs a serious pep talk.</div>", unsafe_allow_html=True)
        st.snow()

    # -------------------------------------------------------------
    # Errors
    # -------------------------------------------------------------
    st.subheader("💥 Prediction Errors")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='stat-card'>MAE<br><span style='font-size:1.6rem;'>{mae:.2f}</span></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='stat-card'>MSE<br><span style='font-size:1.6rem;'>{mse:.2f}</span></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='stat-card'>RMSE<br><span style='font-size:1.6rem;'>{rmse:.2f}</span></div>", unsafe_allow_html=True)

    st.write("")
    fig_err = go.Figure(
        go.Bar(
            x=[str(h) for h in study_hours],
            y=errors,
            marker_color=["#e57373" if v > 0 else ("#66bb6a" if v < 0 else "#bbbbbb") for v in errors],
        )
    )
    fig_err.update_layout(
        title="Over/Underpredicted by Data Point", xaxis_title="Study Hours", yaxis_title="Error", template="plotly_white"
    )
    fig_err.add_hline(y=0, line_color="#888888")
    st.plotly_chart(fig_err, use_container_width=True)

    # -------------------------------------------------------------
    # Best / worst predictions
    # -------------------------------------------------------------
    st.subheader("🏆 Best & Worst Predictions")
    best_idx = int(np.argmin(abs_error))
    worst_idx = int(np.argmax(abs_error))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class='good-card'>
            <b>🏅 BEST — {study_hours[best_idx]:.0f} hrs</b><br>
            Actual: {actual_scores[best_idx]:.0f} · Predicted: {predicted_score[best_idx]:.2f}<br>
            Error: {errors[best_idx]:.2f}
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class='bad-card'>
            <b>💀 WORST — {study_hours[worst_idx]:.0f} hrs</b><br>
            Actual: {actual_scores[worst_idx]:.0f} · Predicted: {predicted_score[worst_idx]:.2f}<br>
            Error: {errors[worst_idx]:.2f}
            </div>""",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------
    # Categories + full table
    # -------------------------------------------------------------
    st.subheader("📋 Prediction Quality Table")

    col1, col2 = st.columns(2)
    excellent_cutoff = col1.slider("Excellent cutoff (abs error ≤)", 0.0, 20.0, 3.0, step=0.5)
    good_cutoff = col2.slider("Good cutoff (abs error ≤)", excellent_cutoff, 30.0, 7.0, step=0.5)

    categories = np.where(abs_error <= excellent_cutoff, "🌟 Excellent", np.where(abs_error <= good_cutoff, "👍 Good", "📚 Needs Work"))
    positive_error = predicted_score > actual_scores
    perfect = predicted_score == actual_scores
    status = np.where(perfect, "🎯 Perfect", np.where(positive_error, "⬆️ Overpredicted", "⬇️ Underpredicted"))

    table = pd.DataFrame(
        {
            "Study Hours": study_hours,
            "Actual Score": actual_scores,
            "Predicted Score": np.round(predicted_score, 2),
            "Abs Error": np.round(abs_error, 2),
            "Status": status,
            "Category": categories,
        }
    )

    def highlight_category(row):
        color = {"🌟 Excellent": "#e2f9ee", "👍 Good": "#fdf0e0", "📚 Needs Work": "#fbe4e4"}[row["Category"]]
        return [f"background-color: {color}; color: #1b2a6b;"] * len(row)

    st.dataframe(table.style.apply(highlight_category, axis=1), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # Predict new scores
    # -------------------------------------------------------------
    st.subheader("🔮 Predict a New Score")

    tab1, tab2 = st.tabs(["🎯 Single value", "🎲 Multiple values"])

    with tab1:
        new_hours = st.number_input("Study hours", min_value=0.0, value=float(stuhours_mean), step=0.5)
        predicted_new_score = slope * new_hours + intercept
        st.markdown(
            f"<div class='stat-card' style='font-size:1.4rem;'>🔮 Predicted score: <span style='color:#1b2a6b; font-size:2rem;'>{predicted_new_score:.2f}</span></div>",
            unsafe_allow_html=True,
        )

    with tab2:
        default_multi = "3,4,5,6,7"
        raw_multi = st.text_input("Study hours (comma-separated)", value=default_multi)
        try:
            new_multiple_hours = np.array([float(x.strip()) for x in raw_multi.split(",") if x.strip() != ""])
            if new_multiple_hours.size == 0:
                st.info("Enter at least one value.")
            else:
                predicted_multiple_score = slope * new_multiple_hours + intercept
                st.dataframe(
                    pd.DataFrame({"Study Hours": new_multiple_hours, "Predicted Score": np.round(predicted_multiple_score, 2)}),
                    use_container_width=True,
                    hide_index=True,
                )
        except ValueError:
            st.error("Please enter only numbers separated by commas.")
else:
    st.info("👆 Hit the crazy button above to fit the model and unleash the visuals.")
