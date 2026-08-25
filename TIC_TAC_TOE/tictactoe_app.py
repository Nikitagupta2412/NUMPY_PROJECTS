import numpy as np
import streamlit as st

st.set_page_config(page_title="Tic Tac Toe", page_icon="✨", layout="centered")

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes pop-in {
        0% { transform: scale(0.3); opacity: 0; }
        70% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes win-glow {
        0%, 100% { box-shadow: 0 0 0 rgba(255, 196, 0, 0.0); }
        50% { box-shadow: 0 0 22px rgba(255, 196, 0, 0.8); }
    }
    div.stButton > button {
        height: 100px;
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        border-radius: 16px;
        border: 2px solid #d9c8f5;
        background: #fbf7ff;
        color: #4b2e83;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease, border 0.15s ease;
    }
    div.stButton > button:hover:not(:disabled) {
        border: 2px solid #b98cf0;
        background: #f3e9ff;
        color: #4b2e83;
        transform: scale(1.05);
        box-shadow: 0 4px 14px rgba(185, 140, 240, 0.35);
    }
    div.stButton > button:active:not(:disabled) {
        transform: scale(0.96);
    }
    div.stButton > button:disabled {
        background: #fbf7ff;
        color: #4b2e83;
        opacity: 1;
        cursor: default;
        animation: pop-in 0.25s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>✨ Tic Tac Toe ✨</h1>", unsafe_allow_html=True)

SYMBOLS = {0: "", 1: "❌", 2: "⭕"}
NAMES = {1: "Player 1 (❌)", 2: "Player 2 (⭕)"}

# ---------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------
if "board" not in st.session_state:
    st.session_state.board = np.zeros((3, 3), dtype=int)
if "current_player" not in st.session_state:
    st.session_state.current_player = 1
if "winner" not in st.session_state:
    st.session_state.winner = None


def reset_game():
    st.session_state.board = np.zeros((3, 3), dtype=int)
    st.session_state.current_player = 1
    st.session_state.winner = None


# ---------------------------------------------------------------------
# Win-checking logic (mirrors the notebook's numpy checks)
# ---------------------------------------------------------------------
def check_winner(board, player):
    main_diag = np.diag(board)
    anti_diag = np.diag(np.fliplr(board))
    return (
        np.all(main_diag == player)
        or np.all(anti_diag == player)
        or np.any(np.all(board == player, axis=1))
        or np.any(np.all(board == player, axis=0))
    )


def get_status():
    board = st.session_state.board
    available_moves = np.argwhere(board == 0)

    if check_winner(board, 1):
        return NAMES[1]
    if check_winner(board, 2):
        return NAMES[2]
    if len(available_moves) == 0:
        return "DRAW"
    return "In progress"


def handle_click(r, c):
    if st.session_state.winner is not None:
        return
    if st.session_state.board[r, c] != 0:
        return
    st.session_state.board[r, c] = st.session_state.current_player
    status = get_status()
    if status != "In progress":
        st.session_state.winner = status
    else:
        st.session_state.current_player = 2 if st.session_state.current_player == 1 else 1


# ---------------------------------------------------------------------
# Turn / status banner
# ---------------------------------------------------------------------
if st.session_state.winner is None:
    st.markdown(
        f"<h3 style='text-align:center;'>Turn: {NAMES[st.session_state.current_player]}</h3>",
        unsafe_allow_html=True,
    )
elif st.session_state.winner == "DRAW":
    st.markdown("<h3 style='text-align:center;'>🤝 It's a draw!</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h3 style='text-align:center;'>🎉 {st.session_state.winner} wins! 🎉</h3>", unsafe_allow_html=True)
    st.balloons()

st.write("")

# ---------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------
board = st.session_state.board
for r in range(3):
    cols = st.columns(3, gap="small")
    for c in range(3):
        with cols[c]:
            st.button(
                SYMBOLS[board[r, c]] or " ",
                key=f"cell_{r}_{c}",
                on_click=handle_click,
                args=(r, c),
                disabled=(board[r, c] != 0 or st.session_state.winner is not None),
                use_container_width=True,
            )

st.write("")

# ---------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.button("🔄 New game", on_click=reset_game, use_container_width=True)

# ---------------------------------------------------------------------
# Raw board view (mirrors the notebook's array output)
# ---------------------------------------------------------------------
with st.expander("Show raw board array"):
    st.write(board)
    st.caption("0 = empty, 1 = Player 1 (X), 2 = Player 2 (O)")
