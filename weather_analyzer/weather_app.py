import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weather Analyzer", page_icon="🌡️", layout="centered")
st.title("🌡️ Weather Analyzer")

# ---------------------------------------------------------------------
# Input: editable temperature matrix (cities x days)
# ---------------------------------------------------------------------
default_data = pd.DataFrame(
    [
        [34, 36, 38, 35, 33, 37, 39],
        [30, 31, 32, 30, 29, 31, 32],
        [32, 33, 35, 34, 31, 33, 36],
    ],
    index=["Delhi", "Mumbai", "Pune"],
    columns=[f"Day {i + 1}" for i in range(7)],
)

st.subheader("Temperature Data (rows = cities, columns = days)")
st.caption("Edit values directly, rename the row index to change city names, or add/remove rows and columns.")

edited_df = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True,
)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

temp_data = edited_df.to_numpy(dtype=float)
cities = list(edited_df.index.astype(str))
day_names = list(edited_df.columns.astype(str))

# ---------------------------------------------------------------------
# Averages
# ---------------------------------------------------------------------
st.header("Averages")

city_avg = np.mean(temp_data, axis=1)
daily_avg = np.mean(temp_data, axis=0)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Average temp per city")
    st.dataframe(
        pd.DataFrame({"City": cities, "Avg Temp": city_avg}).set_index("City"),
        use_container_width=True,
    )
with col2:
    st.subheader("Average temp per day (across cities)")
    st.dataframe(
        pd.DataFrame({"Day": day_names, "Avg Temp": daily_avg}).set_index("Day"),
        use_container_width=True,
    )

st.line_chart(pd.DataFrame(temp_data.T, index=day_names, columns=cities))

# ---------------------------------------------------------------------
# Extremes
# ---------------------------------------------------------------------
st.header("Extremes")

highest_temp = np.max(temp_data)
lowest_temp = np.min(temp_data)
highest_city_idx, highest_day_idx = np.unravel_index(np.argmax(temp_data), temp_data.shape)
lowest_city_idx, lowest_day_idx = np.unravel_index(np.argmin(temp_data), temp_data.shape)

col1, col2 = st.columns(2)
with col1:
    st.metric("Highest temp recorded", f"{highest_temp:.0f}")
    st.caption(f"{cities[highest_city_idx]}, {day_names[highest_day_idx]}")
with col2:
    st.metric("Lowest temp recorded", f"{lowest_temp:.0f}")
    st.caption(f"{cities[lowest_city_idx]}, {day_names[lowest_day_idx]}")

st.info(
    "Note: your original notebook used `np.argmax(temp_data)` directly as a city index, "
    "but that actually returns a flattened position across the whole array. This app uses "
    "`np.unravel_index` to correctly map it back to a city and day."
)

city_highavg_idx = int(np.argmax(city_avg))
st.metric("City with highest average temp", cities[city_highavg_idx], f"{city_avg[city_highavg_idx]:.2f}")

# ---------------------------------------------------------------------
# Per-city max/min
# ---------------------------------------------------------------------
st.header("Per-City Highs and Lows")

high_temp_rec_each_city = np.max(temp_data, axis=1)
low_temp_rec_each_city = np.min(temp_data, axis=1)

st.dataframe(
    pd.DataFrame(
        {"City": cities, "Highest Temp": high_temp_rec_each_city, "Lowest Temp": low_temp_rec_each_city}
    ).set_index("City"),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Extreme heat days
# ---------------------------------------------------------------------
st.header("Extreme Heat Days")

heat_threshold = st.slider("Heat threshold", min_value=int(temp_data.min()), max_value=int(temp_data.max()), value=35)

extreme_heat_readings = temp_data[temp_data > heat_threshold]
st.write(f"**All readings above {heat_threshold}:**", extreme_heat_readings.tolist())
st.write(f"**Total count:** {extreme_heat_readings.size}")

# Correct per-city breakdown (the original notebook's zip mislabeled row/col index arrays as cities)
rows, cols = np.where(temp_data > heat_threshold)
per_city_counts = {city: 0 for city in cities}
for r in rows:
    per_city_counts[cities[r]] += 1

st.subheader(f"Number of days above {heat_threshold}, by city")
st.dataframe(
    pd.DataFrame({"City": list(per_city_counts.keys()), "Days Above Threshold": list(per_city_counts.values())}).set_index("City"),
    use_container_width=True,
)

with st.expander("Show raw (row, day) positions above threshold"):
    positions = [(cities[r], day_names[c], temp_data[r, c]) for r, c in zip(rows, cols)]
    st.dataframe(pd.DataFrame(positions, columns=["City", "Day", "Temp"]), use_container_width=True)
