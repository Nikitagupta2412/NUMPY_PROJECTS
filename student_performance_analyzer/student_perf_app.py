import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Dashboard")

# ---------------------------------------------------------------------
# Input: editable student x subject table
# ---------------------------------------------------------------------
default_data = pd.DataFrame(
    [
        [85, 78, 92, 90],
        [55, 60, 58, 85],
        [95, 92, 98, 95],
        [40, 50, 45, 60],
        [70, 75, 80, 88],
    ],
    index=["Alice", "Bob", "Charlie", "David", "Eva"],
    columns=["python", "maths", "english", "attendance"],
)

st.subheader("Student Data")
st.caption("The first 3 columns are academic subject scores; the last column is attendance %. Edit freely, rename the row index to change student names, or add/remove rows.")

edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

if edited_df.shape[1] < 4:
    st.error("Please keep at least 4 columns: 3 academic subjects + attendance.")
    st.stop()

data = edited_df.to_numpy(dtype=float)
students = np.array(edited_df.index.astype(str))
subjects = np.array(edited_df.columns.astype(str))
n_academic_cols = data.shape[1] - 1  # everything except the last (attendance) column

# ---------------------------------------------------------------------
# Academic averages
# ---------------------------------------------------------------------
st.header("Academic Averages")

academic_avg = np.mean(data[:, 0:n_academic_cols], axis=1)
subject_avg = np.mean(data, axis=0)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Per-student academic average")
    st.dataframe(
        pd.DataFrame({"Student": students, "Academic Avg": academic_avg}).set_index("Student"),
        use_container_width=True,
    )
with col2:
    st.subheader("Per-subject/column average")
    st.dataframe(
        pd.DataFrame({"Column": subjects, "Average": subject_avg}).set_index("Column"),
        use_container_width=True,
    )

st.bar_chart(pd.DataFrame({"Academic Avg": academic_avg}, index=students))

# ---------------------------------------------------------------------
# Top performers
# ---------------------------------------------------------------------
st.header("Top Performers")

top_cutoff = st.slider("Academic average cutoff for 'top performer'", min_value=0, max_value=100, value=80)
top_performers = students[academic_avg >= top_cutoff]

if len(top_performers):
    st.success(f"Top performers (avg ≥ {top_cutoff}): {', '.join(top_performers)}")
else:
    st.info(f"No students have an academic average ≥ {top_cutoff}.")

# ---------------------------------------------------------------------
# Attendance alerts
# ---------------------------------------------------------------------
st.header("Attendance Alerts")

attendance = data[:, -1]
attendance_cutoff = st.slider("Attendance alert threshold (below this = alert)", min_value=0, max_value=100, value=75)
attendance_alert = students[attendance < attendance_cutoff]

st.dataframe(
    pd.DataFrame({"Student": students, "Attendance": attendance}).set_index("Student"),
    use_container_width=True,
)

if len(attendance_alert):
    st.warning(f"Students with attendance below {attendance_cutoff}%: {', '.join(attendance_alert)}")
else:
    st.success(f"No students are below the {attendance_cutoff}% attendance threshold.")

# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------
st.header("Class Ranking")

sorted_idx = np.argsort(academic_avg)[::-1]  # highest academic avg first
rank_table = pd.DataFrame(
    {
        "Rank": range(1, len(sorted_idx) + 1),
        "Student": students[sorted_idx],
        "Academic Avg": academic_avg[sorted_idx],
        "Attendance": attendance[sorted_idx],
    }
).set_index("Rank")

st.dataframe(rank_table, use_container_width=True)
st.caption(
    "Note: your original loop paired each ranked student with `academic_avg[i]`/`attendance[i]` using the "
    "loop index, not the sorted student's own scores — that mismatched names with the wrong numbers once the "
    "order changed. This table pairs each student with their correct scores after sorting."
)

# ---------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------
st.header("Highlights")

highest_subject_idx = int(np.argmax(subject_avg[0:n_academic_cols]))
best_student_idx = int(np.argmax(academic_avg))

col1, col2 = st.columns(2)
col1.metric("Subject with highest average", subjects[highest_subject_idx], f"{subject_avg[highest_subject_idx]:.2f}")
col2.metric("Brightest student (highest academic avg)", students[best_student_idx], f"{academic_avg[best_student_idx]:.2f}")

# ---------------------------------------------------------------------
# Perfect balance
# ---------------------------------------------------------------------
st.header("Perfect Balance: Good Grades + Good Attendance")

perfect_students = students[np.where((academic_avg >= top_cutoff) & (attendance >= attendance_cutoff))]

if len(perfect_students):
    st.success(
        f"Students with academic avg ≥ {top_cutoff} AND attendance ≥ {attendance_cutoff}%: "
        f"{', '.join(perfect_students)}"
    )
else:
    st.info("No students currently meet both the academic and attendance cutoffs.")
