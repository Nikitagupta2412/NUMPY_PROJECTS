import numpy as np
import streamlit as st

st.set_page_config(page_title="Student Marks Analysis", page_icon="📊", layout="centered")
st.title("📊 Student Marks Analysis")

# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------
st.sidebar.header("Input")
default_marks = "98,94,94,83,82,95,83,92,93,84"
raw = st.sidebar.text_area(
    "Marks (comma-separated)",
    value=default_marks,
    height=100,
    help="Enter one score per student, separated by commas.",
)
pass_cutoff = st.sidebar.number_input("Pass mark cutoff", min_value=0, max_value=100, value=85, step=1)
bonus = st.sidebar.number_input("Bonus marks to add (clipped 0-100)", min_value=-100, max_value=100, value=5, step=1)

# Parse marks safely
try:
    marks = np.array([int(x.strip()) for x in raw.split(",") if x.strip() != ""])
    if marks.size == 0:
        st.warning("Enter at least one mark to see the analysis.")
        st.stop()
except ValueError:
    st.error("Please enter only numbers separated by commas (e.g. 98,94,83).")
    st.stop()

# ---------------------------------------------------------------------
# Basic info
# ---------------------------------------------------------------------
st.header("Class Data")
st.write("**Marks array:**", marks.tolist())

col1, col2, col3 = st.columns(3)
col1.metric("Total students", int(marks.size))
col2.metric("Array shape", str(marks.shape))
col3.metric("Class average", f"{np.mean(marks):.2f}")

# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------
st.header("Summary Statistics")

max_idx = int(np.argmax(marks))
min_idx = int(np.argmin(marks))

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏆 Highest")
    st.write(f"**Marks:** {np.max(marks)}")
    st.write(f"**Student index:** {max_idx}")
with col2:
    st.subheader("📉 Lowest")
    st.write(f"**Marks:** {np.min(marks)}")
    st.write(f"**Student index:** {min_idx}")

st.write(f"**Total score by the class:** {int(np.sum(marks))}")

# ---------------------------------------------------------------------
# Pass/fail analysis
# ---------------------------------------------------------------------
st.header("Pass / Fail Analysis")

pass_students = marks[marks >= pass_cutoff]
num_passed = pass_students.size
pass_percentage = (num_passed / marks.size) * 100 if marks.size else 0

col1, col2, col3 = st.columns(3)
col1.metric("Students passed", int(num_passed))
col2.metric("Sum of passing marks", int(np.sum(pass_students)) if num_passed else 0)
col3.metric("Pass %", f"{pass_percentage:.1f}%")

st.write(f"**Marks of students who passed (>= {pass_cutoff}):**", pass_students.tolist())

# ---------------------------------------------------------------------
# Bonus / clipped marks
# ---------------------------------------------------------------------
st.header("Adjusted Marks")
adjusted = np.clip(marks + bonus, a_min=0, a_max=100)
st.write(f"**Marks after adding {bonus} (clipped to 0-100):**", adjusted.tolist())

# ---------------------------------------------------------------------
# Full table
# ---------------------------------------------------------------------
st.header("Per-Student Table")
table = [
    {
        "Student #": i,
        "Marks": int(m),
        "Passed": bool(m >= pass_cutoff),
        "Adjusted Marks": int(a),
    }
    for i, (m, a) in enumerate(zip(marks, adjusted))
]
st.dataframe(table, use_container_width=True)
