import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Ratings Analysis", page_icon="🎬", layout="centered")
st.title("🎬 Movie Ratings Analysis")

# ---------------------------------------------------------------------
# Input: editable ratings matrix (users x movies)
# ---------------------------------------------------------------------
default_ratings = pd.DataFrame(
    [
        [4, 7, 8, 10],
        [3, 9, 10, 3],
        [3, 9, 10, 10],
    ],
    index=[f"User {i + 1}" for i in range(3)],
    columns=[f"Movie {j + 1}" for j in range(4)],
)

st.subheader("Ratings Data (rows = users, columns = movies)")
st.caption("Edit values directly, or use the row/column controls to resize the table.")

edited_df = st.data_editor(
    default_ratings,
    num_rows="dynamic",
    use_container_width=True,
)

if edited_df.empty or edited_df.isnull().values.any():
    st.warning("Please make sure every cell has a numeric value.")
    st.stop()

ratings = edited_df.to_numpy(dtype=float)
user_names = list(edited_df.index.astype(str))
movie_names = list(edited_df.columns.astype(str))

st.write("**Shape:**", ratings.shape)

# ---------------------------------------------------------------------
# Averages
# ---------------------------------------------------------------------
st.header("Averages")

avg_rating_of_movie = np.mean(ratings, axis=0)
avg_rating_by_user = np.mean(ratings, axis=1)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Average rating per movie")
    st.dataframe(
        pd.DataFrame({"Movie": movie_names, "Avg Rating": avg_rating_of_movie}).set_index("Movie"),
        use_container_width=True,
    )
with col2:
    st.subheader("Average rating per user")
    st.dataframe(
        pd.DataFrame({"User": user_names, "Avg Rating": avg_rating_by_user}).set_index("User"),
        use_container_width=True,
    )

st.bar_chart(pd.DataFrame({"Avg Rating": avg_rating_of_movie}, index=movie_names))
st.bar_chart(pd.DataFrame({"Avg Rating": avg_rating_by_user}, index=user_names))

# ---------------------------------------------------------------------
# Top movie / top user
# ---------------------------------------------------------------------
st.header("Top Performers")

high_rating_movie_idx = int(np.argmax(avg_rating_of_movie))
user_high_rating_idx = int(np.argmax(avg_rating_by_user))
low_rating_movie_idx = int(np.argmin(avg_rating_of_movie))

col1, col2, col3 = st.columns(3)
col1.metric("Highest-rated movie", movie_names[high_rating_movie_idx], f"{avg_rating_of_movie[high_rating_movie_idx]:.2f}")
col2.metric("Lowest-rated movie", movie_names[low_rating_movie_idx], f"{avg_rating_of_movie[low_rating_movie_idx]:.2f}")
col3.metric("User who rates highest on average", user_names[user_high_rating_idx], f"{avg_rating_by_user[user_high_rating_idx]:.2f}")

# ---------------------------------------------------------------------
# High-rated ratings filter
# ---------------------------------------------------------------------
st.header("High-Rated Entries")

threshold = st.slider("Rating threshold", min_value=0, max_value=10, value=8)
high_rated_mask = ratings >= threshold
ratings_of_high_rated = ratings[high_rated_mask]

st.write(f"**Individual ratings ≥ {threshold}:**", ratings_of_high_rated.tolist())
st.write(f"**Total count of such ratings:** {len(ratings_of_high_rated)}")

st.dataframe(
    pd.DataFrame(high_rated_mask, index=user_names, columns=movie_names),
    use_container_width=True,
)

# ---------------------------------------------------------------------
# Min/max and normalization
# ---------------------------------------------------------------------
st.header("Normalization")

minimum_rating = np.min(ratings)
maximum_rating = np.max(ratings)

col1, col2 = st.columns(2)
col1.metric("Minimum rating", f"{minimum_rating:.0f}")
col2.metric("Maximum rating", f"{maximum_rating:.0f}")

if maximum_rating == minimum_rating:
    st.warning("All ratings are identical, so normalized values are undefined (division by zero).")
else:
    normalized_ratings = (ratings - minimum_rating) / (maximum_rating - minimum_rating)
    st.write("**Normalized ratings (rounded to 2 decimals):**")
    st.dataframe(
        pd.DataFrame(np.round(normalized_ratings, 2), index=user_names, columns=movie_names),
        use_container_width=True,
    )

# ---------------------------------------------------------------------
# Movies with avg rating above cutoff
# ---------------------------------------------------------------------
st.header("Movies Above a Rating Cutoff")
cutoff = st.slider("Average rating cutoff", min_value=0.0, max_value=10.0, value=8.0, step=0.5)
above_cutoff = [(movie_names[i], avg_rating_of_movie[i]) for i in range(len(movie_names)) if avg_rating_of_movie[i] >= cutoff]

if above_cutoff:
    st.dataframe(
        pd.DataFrame(above_cutoff, columns=["Movie", "Avg Rating"]).set_index("Movie"),
        use_container_width=True,
    )
else:
    st.info(f"No movies have an average rating ≥ {cutoff}.")

# ---------------------------------------------------------------------
# Top-N movies
# ---------------------------------------------------------------------
st.header("Top-Performing Movies")

top_n = st.number_input(
    "How many top movies to show", min_value=1, max_value=len(movie_names), value=min(2, len(movie_names)), step=1
)

sorted_indices = np.argsort(avg_rating_of_movie)
top_indices = sorted_indices[-int(top_n):][::-1]

st.dataframe(
    pd.DataFrame(
        {"Movie": [movie_names[i] for i in top_indices], "Avg Rating": [avg_rating_of_movie[i] for i in top_indices]}
    ).set_index("Movie"),
    use_container_width=True,
)
