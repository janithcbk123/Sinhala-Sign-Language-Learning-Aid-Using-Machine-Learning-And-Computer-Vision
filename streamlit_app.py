import streamlit as st
from bokeh.models.widgets import Div

# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sinhala Sign Language Learning Aid",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------

st.title('📚 Sinhala Sign Language Learning Aid')
st.info('Sinhala Sign Language Learning Aid Using Machine Learning And Computer Vision')

if st.sidebar.button('Github'):
    js = "window.open('https://github.com/janithcbk123/Sinhala-Sign-Language-Learning-Aid-Using-Machine-Learning-And-Computer-Vision')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

# ----------------------------------------------------------------------

st.sidebar.title('Menu')
st.sidebar.subheader('Settings')

# ----------------------------------------------------------------------

app_mode= st.sidebar.selectbox('Choose the App Mode',
                               ['Learn Sign Language','Show Alphabet','About App'])

# ----------------------------------------------------------------------

if app_mode== 'About App':
    st.markdown('Developed by **Thesara_Wiki** & **Janith_C**')
    st.markdown('App Made using **Mediapipe** & **Open CV**')

    st.markdown(
        """
        <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
    width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 350px
    margin-left: -350px
    </style>
        """,unsafe_allow_html=True,)

    st.markdown('''
                # Tutorial \n
                '''
                )
    original_title = '<pre style="font-family:Aku & Kamu; color:#FD0101 ; font-size: 28px;font-weight:Bold">*NOTE</pre>'
    st.markdown(original_title, unsafe_allow_html=True)
    original_title= '''<pre style="font-family:Aku & Kamu; color:#FD0160 ; font-size: 24px;">
    Video Option will Experience Lag in  Browsers.
    If It's <strong>Lagging</strong> just <strong>Reload</strong> & Choose your option <strong>ASAP</strong>  
    Webcam Will Take about <strong>20 Seconds</strong> to Load
    
    Update :
    1) We discovered that you can't use Webcam Online,
    Because then it will try Access Server's Which we don't Own.
    
    2) Hand Marks are not showing online + Video freezes
    
    <strong>Solution :</strong>
    Go to main Streamlit WebApp Code & Run it Locally by typing
    <strong>streamlit run streamlit_app.py</strong>
    </pre>'''
    # st.markdown('''Video Option will Experience **Lag** in **Browsers**. If It's **Lagging** just **Reload** & Choose your option ASAP eg: **Choosing Max Hands** or **Using Webcam**. Webcam Will Take about **20 Seconds** to Load ''')
    st.markdown(original_title, unsafe_allow_html=True)
    
# ----------------------------------------------------------------------



