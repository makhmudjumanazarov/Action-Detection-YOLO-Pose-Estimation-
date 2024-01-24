import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import tempfile

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence = 0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius = 1)

# st.write(""" ### Face Detection """)
         
# Upload a video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
    
    # Open the video file for reading
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if the video file is open
    if cap.isOpened():
        fps = 0
        fpss = []
        prev_time = 0   
        curr_time = 0
        fps_out = st.empty()
        image_out = st.empty()
        
        # Read and display frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            prev_time = time.time()

            frame = cv2.resize(frame, (width, height))

            # Flip the image horizontally for a later selfie-view display    
            # Also convert the color space from BGR to RGB 
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # To improve performance 
            frame.flags.writeable = False

            # Get the result 
            results = face_mesh.process(frame)

            # To improve performance 
            frame.flags.writeable = True

            # Convert the color space from RGB to BGR 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = frame.shape

            face_3d = []
            face_2d = []
            time.sleep(0.1)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 61 or idx == 1 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            
                            x, y = int(lm.x * img_w),  int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])
                    
                    # Convert it to the NumPy array 
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array 
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix 
                    focal_length = 1 * img_w

                    cam_matrix =np.array([ [focal_length, 0, img_h / 2], 
                                          [0, focal_length, img_w / 2], 
                                          [0, 0, 1]])
                    
                    # The distortion parameters 
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP 
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree 
                    x = angles[0] * 360
                    y = angles[1] * 360
                    z = angles[2] * 360 

                    # See where the user's head tilting
                    if y < -5:
                        text = "Looking Left"
                    elif y > 5:
                        text = "Looking Right"
                    elif x < -5:
                        text = "Looking Down" 
                    elif x > 5:
                        text = "Looking Up"
                    else:
                        text = "Forward"
                    
                    # # Display the nose direction
                    # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                    # cv2.line(frame, p1, p2, (255, 0, 0), 3)

                    # # Add the text on the image
                    # cv2.putText(frame, text, (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    # cv2.putText(frame, "x: " + str(np.round(x, 2)), (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                    # cv2.putText(frame, "y: " + str(np.round(y, 2)), (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                    # cv2.putText(frame, "z: " + str(np.round(z, 2)), (1000, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))

                    mp_drawing.draw_landmarks(
                        image = frame, 
                        landmark_list = face_landmarks, 
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = drawing_spec, 
                        connection_drawing_spec = drawing_spec
                    )


            # Display the frame in Streamlit
            image_out.image(frame, channels="BGR", use_column_width=True)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_out.write(f"FPS:{fps}")
            
        # Release everything after the job is finished
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    else:
        st.write("Error: Unable to open the video file.")
else:
    st.write("Please upload a video file to display.")


    