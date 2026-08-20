import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Data Analysis", page_icon="🛒", layout="centered")
st.title("🛒 Sales Data Analysis")

# ---------------------------------------------------------------------
# Input: editable sales matrix (stores x months)
# ---------------------------------------------------------------------
default_sales = pd.DataFrame(
    [
        [10, 12, 15],
        [20, 18, 22],
        [8, 9, 11],
        [14, 15, 17],
    ],
    index=[f"Store {i + 1}" for i in range(4)],
    columns=[f"Month {j + 1}" for j in range(3)],
)

st.subheader("Sales Data (rows = stores, columns = months)")
st.caption("Edit values directly, or use the row/column controls to resize the table.")

edited_df = st.data_editor(
    default_sales,
    num_rows="dynamic",
    use_container_width=True,
)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

sales = edited_df.to_numpy(dtype=float)
store_names = list(edited_df.index.astype(str))
month_names = list(edited_df.columns.astype(str))

# ---------------------------------------------------------------------
# Basic info
# ---------------------------------------------------------------------
st.header("Array Info")
col1, col2 = st.columns(2)
col1.metric("Dimensions (ndim)", sales.ndim)
col2.metric("Shape", str(sales.shape))

# ---------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------
st.header("Totals")

store_total = np.sum(sales, axis=1)
month_total = np.sum(sales, axis=0)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Total sales per store")
    st.dataframe(
        pd.DataFrame({"Store": store_names, "Total": store_total}).set_index("Store"),
        use_container_width=True,
    )
with col2:
    st.subheader("Total sales per month")
    st.dataframe(
        pd.DataFrame({"Month": month_names, "Total": month_total}).set_index("Month"),
        use_container_width=True,
    )

st.bar_chart(pd.DataFrame({"Total": store_total}, index=store_names))
st.bar_chart(pd.DataFrame({"Total": month_total}, index=month_names))

# ---------------------------------------------------------------------
# Best performers
# ---------------------------------------------------------------------
st.header("Best Performers")

best_store_idx = int(np.argmax(store_total))
best_month_idx = int(np.argmax(month_total))

col1, col2 = st.columns(2)
col1.metric("Best-performing store", store_names[best_store_idx], f"{store_total[best_store_idx]:.0f} total")
col2.metric("Best-performing month", month_names[best_month_idx], f"{month_total[best_month_idx]:.0f} total")

# ---------------------------------------------------------------------
# Slice explorer (row/column/single-cell lookups)
# ---------------------------------------------------------------------
st.header("Explore a Slice")
tab1, tab2, tab3 = st.tabs(["By Store", "By Month", "Single Cell"])

with tab1:
    store_choice = st.selectbox("Choose a store", store_names, key="store_slice")
    idx = store_names.index(store_choice)
    st.write(f"**Sales for {store_choice}:**", sales[idx].tolist())

with tab2:
    month_choice = st.selectbox("Choose a month", month_names, key="month_slice")
    jdx = month_names.index(month_choice)
    st.write(f"**Sales for {month_choice} across all stores:**", sales[:, jdx].tolist())

with tab3:
    c1, c2 = st.columns(2)
    store_choice2 = c1.selectbox("Store", store_names, key="cell_store")
    month_choice2 = c2.selectbox("Month", month_names, key="cell_month")
    i2 = store_names.index(store_choice2)
    j2 = month_names.index(month_choice2)
    st.write(f"**Sales of {store_choice2} in {month_choice2}:** {sales[i2, j2]:.0f}")

# ---------------------------------------------------------------------
# Adjusted sales (clip)
# ---------------------------------------------------------------------
st.header("Adjusted Sales")
col1, col2 = st.columns(2)
multiplier = col1.number_input("Multiplier", min_value=0.0, value=1.10, step=0.05)
clip_max = col2.number_input("Clip max", min_value=0.0, value=20.0, step=1.0)

updated_sales = np.clip(sales * multiplier, a_min=0, a_max=clip_max)
st.dataframe(
    pd.DataFrame(updated_sales, index=store_names, columns=month_names),
    use_container_width=True,
)
