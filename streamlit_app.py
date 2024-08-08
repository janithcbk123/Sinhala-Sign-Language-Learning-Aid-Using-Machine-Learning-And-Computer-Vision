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

    # 2. Prediction logic
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))


        # 3. Viz logic
        if np.unique(predictions[-10:])[0]==np.argmax(res):
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

    cv2.rectangle(image, (0 ,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3 ,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)




    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    
else:
    st.write('Stopped')


st.camera_input("das 4")
