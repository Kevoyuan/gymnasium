import streamlit as st
from pages import acrobot_page

PAGES = {
    "Acrobot Environment": acrobot_page,
    # "Another RL Example": another_rl_example,
}

# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# page = PAGES[selection]
# page.app()
