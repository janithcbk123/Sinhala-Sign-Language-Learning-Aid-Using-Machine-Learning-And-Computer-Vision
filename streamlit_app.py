import streamlit as st


# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sinhala Sign Language Learning Aid",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------

if st.sidebar.button('Github'):
    js = "window.open('https://github.com/shashankanand13monu/Game-Automation')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

# ----------------------------------------------------------------------


st.title('📚 Sinhala Sign Language Learning Aid')
st.info('Sinhala Sign Language Learning Aid Using Machine Learning And Computer Vision')
