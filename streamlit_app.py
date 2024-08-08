import cv2
import streamlit as st

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    # Read feed
    ret, frame = cap.read()
    
    # Make detections
    image, results = mediapipe_detection(frame, holistic)
    #print(results)

    # Draw landmarks
    draw_styled_landmarks(image, results)


    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
else:
    st.write('Stopped')


st.camera_input("das 3")
