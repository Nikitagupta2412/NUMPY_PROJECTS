import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Expense Data Analysis", page_icon="💰", layout="centered")
st.title("💰 Weekly Expense Analysis")

# ---------------------------------------------------------------------
# Input: editable expense matrix (days x categories)
# ---------------------------------------------------------------------
default_data = pd.DataFrame(
    [
        [23, 443, 384],
        [73, 94, 934],
        [67, 4903, 4390],
        [783, 4, 43],
        [83, 409, 93],
        [3, 442, 23],
        [33, 43, 23],
    ],
    index=[f"Day {i + 1}" for i in range(7)],
    columns=["Food", "Transport", "Other"],
)

st.subheader("Expense Data (rows = days)")
st.caption("Edit values directly, rename columns to match your real categories, or add/remove days.")

edited_df = st.data_editor(
    default_data,
    num_rows="dynamic",
    use_container_width=True,
)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

expense_data = edited_df.to_numpy(dtype=float)
day_names = list(edited_df.index.astype(str))
category_names = list(edited_df.columns.astype(str))

# ---------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------
st.header("Totals")

total_spent = np.sum(expense_data)
category_totals = np.sum(expense_data, axis=0)
daily_totals = np.sum(expense_data, axis=1)

st.metric("Total weekly expenditure", f"{total_spent:,.0f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Category-wise total")
    st.dataframe(
        pd.DataFrame({"Category": category_names, "Total": category_totals}).set_index("Category"),
        use_container_width=True,
    )
with col2:
    st.subheader("Daily total")
    st.dataframe(
        pd.DataFrame({"Day": day_names, "Total": daily_totals}).set_index("Day"),
        use_container_width=True,
    )

st.bar_chart(pd.DataFrame({"Daily Total": daily_totals}, index=day_names))

# ---------------------------------------------------------------------
# Expensive days
# ---------------------------------------------------------------------
st.header("Days With High Spending")

spend_threshold = st.slider("Threshold", min_value=0, max_value=int(daily_totals.max()) + 100, value=500, step=50)
expensive_mask = daily_totals >= spend_threshold
expensive_days = [day_names[i] for i in range(len(day_names)) if expensive_mask[i]]

if expensive_days:
    st.write(f"**Days with total spending ≥ {spend_threshold}:**", expensive_days)
else:
    st.info(f"No days had spending ≥ {spend_threshold}.")

# ---------------------------------------------------------------------
# Category spotlights
# ---------------------------------------------------------------------
st.header("Category Spotlights")

food_idx = 0 if "Food" not in category_names else category_names.index("Food")
transport_idx = 1 if "Transport" not in category_names else category_names.index("Transport")

col1, col2 = st.columns(2)
with col1:
    highest_val = np.max(expense_data[:, food_idx])
    highest_day = day_names[int(np.argmax(expense_data[:, food_idx]))]
    st.metric(f"Highest {category_names[food_idx]} expense", f"{highest_val:.0f}", highest_day)
with col2:
    avg_val = np.mean(expense_data[:, transport_idx])
    st.metric(f"Average daily {category_names[transport_idx]} spend", f"{avg_val:.2f}")

# ---------------------------------------------------------------------
# Costliest / cheapest day
# ---------------------------------------------------------------------
st.header("Costliest and Cheapest Days")

costliest_idx = int(np.argmax(daily_totals))
cheapest_idx = int(np.argmin(daily_totals))
top_expense_category_idx = int(np.argmax(category_totals))

col1, col2, col3 = st.columns(3)
col1.metric("Costliest day", day_names[costliest_idx], f"{daily_totals[costliest_idx]:.0f}")
col2.metric("Cheapest day", day_names[cheapest_idx], f"{daily_totals[cheapest_idx]:.0f}")
col3.metric("Top spending category", category_names[top_expense_category_idx], f"{category_totals[top_expense_category_idx]:.0f}")

# ---------------------------------------------------------------------
# Top-N most expensive days
# ---------------------------------------------------------------------
st.header("Top Spending Days")

top_n = st.number_input("How many top days to show", min_value=1, max_value=len(day_names), value=min(3, len(day_names)), step=1)
sorted_indices = np.argsort(daily_totals)
top_indices = sorted_indices[-int(top_n):][::-1]

st.dataframe(
    pd.DataFrame({"Day": [day_names[i] for i in top_indices], "Total Spent": [daily_totals[i] for i in top_indices]}).set_index("Day"),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Spending breakdown (pie chart)
# ---------------------------------------------------------------------
st.header("Spending Breakdown by Category")

piechart_pct = (category_totals / total_spent) * 100 if total_spent else np.zeros_like(category_totals)

fig = px.pie(
    names=category_names,
    values=piechart_pct,
    title="Share of Total Spending by Category",
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    pd.DataFrame({"Category": category_names, "% of Total": np.round(piechart_pct, 2)}).set_index("Category"),
    use_container_width=True,
)
