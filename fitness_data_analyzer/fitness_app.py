import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fitness Data Analysis", page_icon="🏃", layout="centered")
st.title("🏃 Fitness Data Analysis")

# ---------------------------------------------------------------------
# Input: editable fitness matrix (days x metrics)
# ---------------------------------------------------------------------
default_data = pd.DataFrame(
    [
        [8000, 2200, 45],
        [12000, 2800, 60],
        [4500, 1600, 20],
        [10500, 2500, 50],
        [3000, 1400, 15],
    ],
    index=[f"Day {i + 1}" for i in range(5)],
    columns=["Steps", "Calories Burned", "Active Minutes"],
)

st.subheader("Fitness Data (rows = days)")
st.caption("Edit values directly, or use the row controls to add/remove days.")

edited_df = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True,
)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

fitness_data = edited_df.to_numpy(dtype=float)
day_names = list(edited_df.index.astype(str))
metric_names = list(edited_df.columns.astype(str))

st.write("**Shape:**", fitness_data.shape)

# ---------------------------------------------------------------------
# Average metrics
# ---------------------------------------------------------------------
st.header("Average Metrics")

avg_metrics = np.mean(fitness_data, axis=0)
st.dataframe(
    pd.DataFrame({"Metric": metric_names, "Average": avg_metrics}).set_index("Metric"),
    use_container_width=True,
)
st.bar_chart(pd.DataFrame({"Average": avg_metrics}, index=metric_names))

# ---------------------------------------------------------------------
# High step days
# ---------------------------------------------------------------------
st.header("High-Step Days")

step_threshold = st.slider("Step threshold", min_value=0, max_value=20000, value=8000, step=500)
high_step_mask = fitness_data[:, 0] > step_threshold

st.dataframe(
    pd.DataFrame({"Day": day_names, "Steps": fitness_data[:, 0], "Above Threshold": high_step_mask}).set_index("Day"),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Activity status
# ---------------------------------------------------------------------
st.header("Activity Status")

active_threshold = st.slider("Steps needed to be considered 'Active'", min_value=0, max_value=20000, value=2000, step=100)
activity_status = np.where(fitness_data[:, 0] >= active_threshold, "Active", "Rest")

st.dataframe(
    pd.DataFrame({"Day": day_names, "Steps": fitness_data[:, 0], "Status": activity_status}).set_index("Day"),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Min per metric
# ---------------------------------------------------------------------
st.header("Minimums Per Metric")

min_steps = np.min(fitness_data[:, 0])
min_calories = np.min(fitness_data[:, 1])
min_active_minutes = np.min(fitness_data[:, 2])

col1, col2, col3 = st.columns(3)
col1.metric(f"Min {metric_names[0]}", f"{min_steps:.0f}")
col2.metric(f"Min {metric_names[1]}", f"{min_calories:.0f}")
col3.metric(f"Min {metric_names[2]}", f"{min_active_minutes:.0f}")

# ---------------------------------------------------------------------
# Overall min/max and scaled data
# ---------------------------------------------------------------------
st.header("Overall Min/Max and Scaled Data")

metric_min = np.min(fitness_data)
metric_max = np.max(fitness_data)

col1, col2 = st.columns(2)
col1.metric("Overall minimum (across all values)", f"{metric_min:.0f}")
col2.metric("Overall maximum (across all values)", f"{metric_max:.0f}")

if metric_max == metric_min:
    st.warning("All values are identical, so scaled data is undefined (division by zero).")
else:
    scaled_data = (fitness_data - metric_min) / (metric_max - metric_min)
    st.caption("Note: this scales every cell using one global min/max, so columns with larger ranges (like Steps) will dominate.")
    st.dataframe(
        pd.DataFrame(np.round(scaled_data, 3), index=day_names, columns=metric_names),
        use_container_width=True,
    )

# ---------------------------------------------------------------------
# Notable days
# ---------------------------------------------------------------------
st.header("Notable Days")

high_step_day_idx = int(np.argmax(fitness_data[:, 0]))
lowest_col3_day_idx = int(np.argmin(fitness_data[:, 2]))

col1, col2 = st.columns(2)
col1.metric("Highest-step day", day_names[high_step_day_idx], f"{fitness_data[high_step_day_idx, 0]:.0f} steps")
col2.metric(f"Lowest {metric_names[2]} day", day_names[lowest_col3_day_idx], f"{fitness_data[lowest_col3_day_idx, 2]:.0f}")
