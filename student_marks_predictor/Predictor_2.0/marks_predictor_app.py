import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Marks Predictor", page_icon="🎯", layout="centered")

st.markdown(
    """
    <style>
    .big-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7c6ee0, #4fb0a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div.stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        background: linear-gradient(90deg, #7c6ee0, #4fb0a5);
        color: white;
        border: none;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(124, 110, 224, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<p class='big-title'>🎯 Marks Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict a student's score from study hours, attendance & practice test score</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Historical training data (editable, tucked in an expander)
# ---------------------------------------------------------------------
default_students = ["nikita", "javid", "yuvraj", "kirti", "sanik", "tulip", "kartik", "owl"]
default_data = pd.DataFrame(
    [
        [6, 78, 98],
        [2, 65, 99],
        [3, 34, 65],
        [7, 34, 98],
        [8, 67, 87],
        [9, 23, 76],
        [4, 14, 76],
        [10, 76, 98],
    ],
    index=default_students,
    columns=["Study Hours", "Attendance", "Practice Test Score"],
)

with st.expander("📊 Historical student data (used to normalize new inputs)"):
    edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
    st.caption("New inputs are scaled relative to the min/max of this data, so values outside this range will extrapolate.")

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell in the historical data has a numeric value.")
    st.stop()

student_data = edited_df.to_numpy(dtype=float)
hist_students = np.array(edited_df.index.astype(str))
feature_names = list(edited_df.columns)

with st.expander("⚖️ Feature weights"):
    wcols = st.columns(3)
    weights = np.array(
        [
            wcols[0].number_input(f"Weight: {feature_names[0]}", value=1.5, step=0.1),
            wcols[1].number_input(f"Weight: {feature_names[1]}", value=0.3, step=0.1),
            wcols[2].number_input(f"Weight: {feature_names[2]}", value=0.6, step=0.1),
        ]
    )

column_min = np.min(student_data, axis=0)
column_max = np.max(student_data, axis=0)
denom = np.where(column_max - column_min == 0, 1, column_max - column_min)
max_weighted_score = np.sum(weights)  # normalized features max out at 1.0 each

# ---------------------------------------------------------------------
# New student input
# ---------------------------------------------------------------------
st.subheader("Enter a Student's Details")

col1, col2, col3 = st.columns(3)
study_hours = col1.slider(
    feature_names[0], min_value=0.0, max_value=max(24.0, float(column_max[0])), value=float(np.mean(student_data[:, 0])), step=0.5
)
attendance = col2.slider(
    feature_names[1], min_value=0.0, max_value=max(100.0, float(column_max[1])), value=float(np.mean(student_data[:, 1])), step=1.0
)
practice_score = col3.slider(
    feature_names[2], min_value=0.0, max_value=max(100.0, float(column_max[2])), value=float(np.mean(student_data[:, 2])), step=1.0
)

predict_clicked = st.button("🔮 Predict My Score")

# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if predict_clicked or "last_prediction" in st.session_state:
    new_data = np.array([study_hours, attendance, practice_score])
    normalized_input = (new_data - column_min) / denom

    raw_weighted_score = normalized_input @ weights  # scale: 0 to sum(weights)
    final_score = np.clip((raw_weighted_score / max_weighted_score) * 100, 0.0, 100.0)

    st.session_state.last_prediction = {
        "score": final_score,
        "raw_weighted": raw_weighted_score,
    }

    st.info(
        "Note: your original notebook clipped the raw weighted score (max possible ≈ "
        f"{max_weighted_score:.2f}) directly to a 0–100 range, which does nothing useful since that "
        "raw score never gets anywhere near 100 — it just prints a tiny number like 1.74 as if it were "
        "a mark out of 100. This app rescales it to a proper 0–100 percentage first."
    )

    # -------------------------------------------------------------
    # Gauge
    # -------------------------------------------------------------
    if final_score >= 80:
        band_color = "#4fb0a5"
        verdict = "🌟 High predicted score!"
    elif final_score >= 60:
        band_color = "#f2a65a"
        verdict = "👍 Average predicted score."
    else:
        band_color = "#e57373"
        verdict = "📚 Low predicted score — more prep may help."

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=final_score,
            number={"suffix": " / 100", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": band_color},
                "steps": [
                    {"range": [0, 60], "color": "#fbe4e4"},
                    {"range": [60, 80], "color": "#fdf0e0"},
                    {"range": [80, 100], "color": "#e2f2ef"},
                ],
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown(f"<h3 style='text-align:center;'>{verdict}</h3>", unsafe_allow_html=True)

    if final_score >= 80:
        st.balloons()

    # -------------------------------------------------------------
    # Compare against the class
    # -------------------------------------------------------------
    st.subheader("How does this compare to the class?")

    normalized_data = (student_data - column_min) / denom
    class_raw_scores = normalized_data @ weights
    class_final_scores = np.clip((class_raw_scores / max_weighted_score) * 100, 0.0, 100.0)

    all_names = list(hist_students) + ["You"]
    all_scores = list(class_final_scores) + [final_score]
    colors = ["#7c6ee0"] * len(hist_students) + ["#4fb0a5"]

    fig = go.Figure(go.Bar(x=all_names, y=all_scores, marker_color=colors))
    fig.update_layout(title="Predicted Score: You vs the Class", yaxis_title="Predicted Score (0-100)")
    st.plotly_chart(fig, use_container_width=True)

    percentile = float(np.mean(class_final_scores < final_score) * 100)
    st.metric("You're scoring higher than", f"{percentile:.0f}% of the class")
