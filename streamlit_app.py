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

    st.set_option('deprecation.showfileUploaderEncoding',False)
    use_webcam = st.sidebar.button('Use Webcam')
    record= st.sidebar.checkbox("Record Video")
    
    if record:
        st.checkbox("Recording",value=True)
    
    
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

    
    max_hands= st.sidebar.number_input('Maximum Number of Hand',value=1,min_value=1,max_value=4)
    detection_confidence= st.sidebar.slider('Detection Confidence',min_value=0.0,max_value=1.0,value=0.5)
    tracking_confidence= st.sidebar.slider('Tracking Confidence Confidence',min_value=0.0,max_value=1.0,value=0.5)
    st.sidebar.markdown('---')
    
    st.subheader("Input Video")    
    
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    #We get our input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    
    #Recording Part
    codec = cv2.VideoWriter_fourcc('V', 'P', '0','9')
    out= cv2.VideoWriter('output.mp4',codec,fps_input,(width,height))
    
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
     
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Frame Rate</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi1_text = st.markdown ("0")
    with kpi2:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Detected Hands</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi2_text = st.markdown ("0")
    with kpi3:
        original_title = '<p style="text-align: center; font-size: 20px;"><strong>Video Width</strong></p>'
        st.markdown(original_title, unsafe_allow_html=True)
        kpi3_text = st.markdown("0")
    st.markdown ("<hr/>", unsafe_allow_html=True)
    st.subheader('Reload , if webpage hangs')
    st.markdown('---')
    st.subheader("Video Hangs in Browser works fine Locally like this : ")   
    data= 'sample.mp4'
    dat2= 'https://youtu.be/UT7gjebls4A'
    st.video(data, format="video/mp4", start_time=0)
    st.video(dat2)
    # video_file = open('sample.mp4', 'rb')
    # video_bytes = video_file.read()
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
    with mp_hand.Hands(max_num_hands=max_hands,min_detection_confidence=detection_confidence,
                       min_tracking_confidence=tracking_confidence) as hands:
    
        
        while vid.isOpened():
            
            i +=1
            ret, image = vid.read()
            if not ret:
                continue
        
          
            image.flags.writeable=False
            results= hands.process(image)
            image.flags.writeable=True
            image= cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            lmList=[]
            lmList2forModel=[]
            hand_count=0
            if results.multi_hand_landmarks:
                
                for hand_landmark in results.multi_hand_landmarks:
                    hand_count += 1
                    myHands=results.multi_hand_landmarks[0]
                    for id,lm in enumerate(myHands.landmark):
                        h,w,c=image.shape
                        cx,cy=int(lm.x*w), int(lm.y*h)
                        lmList.append([id,cx,cy])
                        lmList2forModel.append([cx,cy])
                    
                    if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                        fingers.append(1)

                    else:
                        fingers.append(0)


                    for id in range(1,5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id]-1][2]:
                            fingers.append(1)


                        else:
                            fingers.append(0)

                    total= fingers.count(1)
                    if total==5:
                        sh= "Acclerate"
                        draw(sh)
                    if total==2 or total==3:
                        sh= "Left"
                        draw(sh)
                    if total==4:
                        sh= "Right"
                        draw(sh)
                    if total==0:
                        sh= "Brake"
                        draw(sh)
                    
                    mp_draw.draw_landmarks(image,hand_landmark,mp_hand.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))
                    
                #FPS Counter Logic
            currTime = time.time()
            fps = 1/ (currTime - prevTime)
            prevTime = currTime
            fingers=[]
            
            if record:
                out.write(image)
            image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            kpi1_text.write(f"<h1 style='text-align: center; color:red; '>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red; '>{hand_count}</h1>", unsafe_allow_html=True)
            
            kpi3_text.write(f"<h1 style='text-align: center; color:red; '>{width}</h1>", unsafe_allow_html=True)
            
            image = cv2.resize(image, (0,0), fx = 0.8, fy =0.8)
            image = image_resize(image = image, width = 320,height=360)
            stframe.image(image, channels = 'BGR', use_column_width=False)
    st.subheader('Output Image')
    
    # st.video('streamlit-st2-2022-01-11-23-01-57.webm')
    # sample= 'streamlit-st2-2022-01-11-23-01-57.webm'
    # sampl= sample.read()
    # st.video(sampl)
    st.text('Video Processed')
    output_video = open('output1.mp4','rb')
    out_bytes= output_video.read()
    st.video(out_bytes)
    

    st.video(video_bytes) 
    vid.release()
    out.release()

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



