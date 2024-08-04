import streamlit as st
from bokeh.themes import theme
from bokeh.models.widgets import Div

# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sinhala Sign Language Learning Aid",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------

# ----------------------------------------------------------------------
if st.sidebar.button('Github'):
    js = "window.open('janithcbk123/Sinhala-Sign-Language-Learning-Aid-Using-Machine-Learning-And-Computer-Vision')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

# ----------------------------------------------------------------------



st.info('Sinhala Sign Language Learning Aid Using Machine Learning And Computer Vision')
