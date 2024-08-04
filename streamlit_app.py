import streamlit as st
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Sinhala Sign Language Learning Aid",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_draw= mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_draw= mp.solutions.drawing_utils
mp_hand= mp.solutions.hands
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# ----------------------------------------------------------------------

st.title('📚 Sinhala Sign Language Learning Aid')
st.info('Sinhala Sign Language Learning Aid Using Machine Learning And Computer Vision')

if st.sidebar.button('Github'):
    js = "window.open('https://github.com/janithcbk123/Sinhala-Sign-Language-Learning-Aid-Using-Machine-Learning-And-Computer-Vision')"  # New tab or window
    # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)

# ----------------------------------------------------------------------

st.sidebar.title('Menu')
st.sidebar.subheader('Settings')

# ----------------------------------------------------------------------

app_mode= st.sidebar.selectbox('Choose the App Mode',
                               ['Learn Sign Language','Show Alphabet','About App'])

# ----------------------------------------------------------------------

if app_mode== 'Learn Sign Language':
    st.markdown('Developed by **Thesara_Wiki** & **Janith_C**')
    st.markdown('App Made using **Mediapipe** & **Open CV**')

    st.sidebar.subheader('Number you want to learn')
    number = st.sidebar.number_input(
    "Insert a number", min_value=1, max_value=30, value=None, step=1, placeholder="Type a number..."
    )

#####################################################################################################################################################################

    colors = [(245,117,16), (117,245,16), (16,117,245)]
    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        return output_frame


    plt.figure(figsize=(18,18))
    plt.imshow(prob_viz(res, actions, image, colors))

    sequence.reverse()

    len(sequence)

    sequence.append('def')

    sequence.reverse()

    sequence[-30:]

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
        
            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
#           sequence.insert(0,keypoints)
#           sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
            
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()

#####################################################################################################################################################################
    
# ----------------------------------------------------------------------

elif app_mode== 'About App':
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



