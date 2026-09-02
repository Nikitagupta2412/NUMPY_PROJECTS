import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Score Prediction Analysis", page_icon="📈", layout="centered")
st.title("📈 Student Score Prediction Analysis")

st.caption(
    "A weighted-sum model predicts each student's score from 3 feature columns, compares it to their "
    "actual score, and measures error — once on raw values, and once on min-max normalized values."
)

# ---------------------------------------------------------------------
# Input: editable student features + actual scores
# ---------------------------------------------------------------------
default_students = ["nikita", "javid", "yuvraj", "kirti", "sanik", "tulip", "kartik", "owl"]
default_features = pd.DataFrame(
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
    columns=["Feature 1", "Feature 2", "Feature 3"],
)
default_features["Actual Score"] = [98, 96, 93, 94, 95, 99, 100, 23]

st.subheader("Student Data")
st.caption("First 3 columns are the predictor features; the last column is the actual score to compare against. Edit freely.")

edited_df = st.data_editor(default_features, num_rows="dynamic", use_container_width=True)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

if edited_df.shape[1] < 4:
    st.error("Please keep at least 4 columns: 3 features + Actual Score.")
    st.stop()

students = np.array(edited_df.index.astype(str))
student_data = edited_df.iloc[:, :-1].to_numpy(dtype=float)
actual_scores = edited_df.iloc[:, -1].to_numpy(dtype=float)
feature_names = list(edited_df.columns[:-1])

st.subheader("Feature Weights")
weight_cols = st.columns(len(feature_names))
weights = np.array(
    [
        weight_cols[i].number_input(f"Weight: {feature_names[i]}", value=[1.5, 0.3, 0.6][i] if i < 3 else 1.0, step=0.1)
        for i in range(len(feature_names))
    ]
)

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
st.header("Normalized Data")

column_min = np.min(student_data, axis=0)
column_max = np.max(student_data, axis=0)
denom = np.where(column_max - column_min == 0, 1, column_max - column_min)
normalized_data = (student_data - column_min) / denom

st.dataframe(
    pd.DataFrame(np.round(normalized_data, 3), index=students, columns=feature_names),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------
st.header("Predicted Scores")

raw_predicted_scores = student_data @ weights
normalized_predicted_scores = normalized_data @ weights

pred_table = pd.DataFrame(
    {
        "Student": students,
        "Actual": actual_scores,
        "Raw Predicted": raw_predicted_scores,
        "Normalized Predicted": normalized_predicted_scores,
    }
).set_index("Student")
st.dataframe(pred_table.round(2), use_container_width=True)

st.caption(
    "The raw model's predictions land on roughly the same scale as the actual scores, but the normalized "
    "model's predictions are on a much smaller scale (features are squeezed to 0–1 before weighting), so "
    "the two are shown as separate charts rather than one shared-scale chart."
)

fig_raw = go.Figure()
fig_raw.add_bar(name="Actual", x=students, y=actual_scores, marker_color="#7c6ee0")
fig_raw.add_bar(name="Raw Predicted", x=students, y=raw_predicted_scores, marker_color="#f2a65a")
fig_raw.update_layout(barmode="group", title="Actual vs Raw Predicted Score", yaxis_title="Score")
st.plotly_chart(fig_raw, use_container_width=True)

fig_norm = go.Figure()
fig_norm.add_bar(name="Normalized Predicted", x=students, y=normalized_predicted_scores, marker_color="#4fb0a5")
fig_norm.update_layout(title="Normalized Predicted Score (own scale)", yaxis_title="Weighted normalized score")
st.plotly_chart(fig_norm, use_container_width=True)

# ---------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------
st.header("Prediction Errors")

raw_error = actual_scores - raw_predicted_scores
normalized_error = actual_scores - normalized_predicted_scores
abs_error = np.abs(raw_error)
normalized_abs_error = np.abs(normalized_error)

raw_avg_error = np.mean(abs_error)
normalized_avg_error = np.mean(normalized_abs_error)

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error (raw)", f"{raw_avg_error:.2f}")
col2.metric("Mean Absolute Error (normalized)", f"{normalized_avg_error:.2f}")

st.dataframe(
    pd.DataFrame(
        {
            "Student": students,
            "Raw Error": raw_error,
            "Abs Raw Error": abs_error,
            "Normalized Error": normalized_error,
            "Abs Normalized Error": normalized_abs_error,
        }
    ).set_index("Student").round(2),
    use_container_width=True,
)

st.caption("Positive error = model underpredicted that student; negative error = model overpredicted.")
fig_err = go.Figure()
fig_err.add_bar(
    x=students,
    y=raw_error,
    marker_color=["#e57373" if v < 0 else "#66bb6a" for v in raw_error],
)
fig_err.update_layout(title="Raw Model Error by Student (Actual − Predicted)", yaxis_title="Error")
fig_err.add_hline(y=0, line_color="#888888")
st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------------------
# Best / worst predictions
# ---------------------------------------------------------------------
st.header("Best & Worst Predictions")

best_idx = int(np.argmin(abs_error))
worst_idx = int(np.argmax(abs_error))
normalized_best_idx = int(np.argmin(normalized_abs_error))
normalized_worst_idx = int(np.argmax(normalized_abs_error))

st.info(
    "Note: in the original script, the normalized best/worst prints reused the raw model's index and "
    "some raw error values by mistake. This app looks each metric up using its own matching index."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw model")
    st.write(f"**Closest prediction:** {students[best_idx]}")
    st.caption(f"Actual: {actual_scores[best_idx]:.0f} · Predicted: {raw_predicted_scores[best_idx]:.2f} · Abs error: {abs_error[best_idx]:.2f}")
    st.write(f"**Farthest prediction:** {students[worst_idx]}")
    st.caption(f"Actual: {actual_scores[worst_idx]:.0f} · Predicted: {raw_predicted_scores[worst_idx]:.2f} · Abs error: {abs_error[worst_idx]:.2f}")
with col2:
    st.subheader("Normalized model")
    st.write(f"**Closest prediction:** {students[normalized_best_idx]}")
    st.caption(f"Actual: {actual_scores[normalized_best_idx]:.0f} · Predicted: {normalized_predicted_scores[normalized_best_idx]:.2f} · Abs error: {normalized_abs_error[normalized_best_idx]:.2f}")
    st.write(f"**Farthest prediction:** {students[normalized_worst_idx]}")
    st.caption(f"Actual: {actual_scores[normalized_worst_idx]:.0f} · Predicted: {normalized_predicted_scores[normalized_worst_idx]:.2f} · Abs error: {normalized_abs_error[normalized_worst_idx]:.2f}")

# ---------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------
st.header("Score Categories")

col1, col2 = st.columns(2)
high_cutoff = col1.slider("High cutoff", min_value=0, max_value=150, value=80)
avg_cutoff = col2.slider("Average cutoff", min_value=0, max_value=150, value=60)

basis = st.radio("Categorize using:", ["Normalized predicted scores", "Raw predicted scores"], horizontal=True)
scores_for_cat = normalized_predicted_scores if basis == "Normalized predicted scores" else raw_predicted_scores

categorize = np.where(scores_for_cat >= high_cutoff, "High", np.where(scores_for_cat >= avg_cutoff, "Average", "Low"))

st.dataframe(
    pd.DataFrame({"Student": students, "Predicted Score": scores_for_cat, "Category": categorize}).set_index("Student").round(2),
    use_container_width=True,
)

total = len(students)
high_count = int(np.sum(scores_for_cat >= high_cutoff))
low_count = int(np.sum(scores_for_cat < avg_cutoff))
high_pct = (np.sum(categorize == "High") / total) * 100
avg_pct = (np.sum(categorize == "Average") / total) * 100
low_pct = (np.sum(categorize == "Low") / total) * 100

col1, col2, col3 = st.columns(3)
col1.metric("High", f"{high_count} ({high_pct:.1f}%)")
col2.metric("Average", f"{total - high_count - low_count} ({avg_pct:.1f}%)")
col3.metric("Low", f"{low_count} ({low_pct:.1f}%)")

# ---------------------------------------------------------------------
# Model bias
# ---------------------------------------------------------------------
st.header("Is the Model Hyping or Underhyping?")

avg_actual_score = np.mean(actual_scores)
avg_raw_predicted_score = np.mean(raw_predicted_scores)
avg_normalized_predicted_score = np.mean(normalized_predicted_scores)

col1, col2, col3 = st.columns(3)
col1.metric("Avg actual score", f"{avg_actual_score:.2f}")
col2.metric("Avg raw predicted", f"{avg_raw_predicted_score:.2f}")
col3.metric("Avg normalized predicted", f"{avg_normalized_predicted_score:.2f}")

col1, col2 = st.columns(2)
with col1:
    verdict = "📈 The raw model is hyping (overpredicting) on average." if avg_raw_predicted_score > avg_actual_score else "📉 The raw model is underhyping (underpredicting) on average."
    st.write(verdict)
with col2:
    verdict = "📈 The normalized model is hyping (overpredicting) on average." if avg_normalized_predicted_score > avg_actual_score else "📉 The normalized model is underhyping (underpredicting) on average."
    st.write(verdict)

# ---------------------------------------------------------------------
# Extra stats
# ---------------------------------------------------------------------
st.header("Extra Stats")

study_hour_min = np.min(normalized_data[:, 0])
st.metric(f"Min normalized value of '{feature_names[0]}'", f"{study_hour_min:.3f}")

improvement = abs_error - normalized_abs_error
most_improved_idx = int(np.argmax(improvement))
st.metric(
    "Most improved student after normalization",
    students[most_improved_idx],
    f"Error reduced by {improvement[most_improved_idx]:.2f}",
)
