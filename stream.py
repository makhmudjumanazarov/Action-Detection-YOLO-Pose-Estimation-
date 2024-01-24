import streamlit as st
from pose import *
import cv2
import tempfile
import time
import numpy as np 

# # Function to initialize video capture
cap  = cv2.VideoCapture()
cap.open("rtsp://admin:1234567q@10.10.0.108:73/h264/ch1/main/av_stream/")

# Upload a video file
# cap = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])


if cap is not None:
    # # Create a temporary file to store the uploaded video
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
    #     tfile.write(cap.read())
    
    # # Open the video file for reading
    # cap = cv2.VideoCapture(tfile.name)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if the video file is open
    if cap.isOpened():
        fps = 0
        prev_time = 0
        curr_time = 0
        summary = []
        vaqt = 0
        fps_out = st.empty()
        image_out = st.empty()
        boshlanishi = time.time()
        sanoq = st.empty()
        total_points = st.empty()
        frames_soni = 0

        # Read and display frames from the video
        while True:
            ret, frame = cap.read()

            if ret:
                frames_soni += 1
            if not ret:
                break
            prev_time = time.time()
            # frame = cv2.resize(frame, (width, height))

            # crop_frame = frame[800:1440, 400:1500]
            # crop_frame = frame
            # crop_frame = frame[100:400, 400:800] # Sadriddin
            crop_frame = frame[400:1000, 500:1000]

            natija, a = predict_pose(crop_frame)
            # time.sleep(0.0099)

            # Display the frame in Streamlit
            image_out.image(np.array(natija), channels="BGR", use_column_width=True)
 
            curr_time = time.time()
            doimiy = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_out.write(f"FPS:{fps}")
            sanoq.write(f"time: {int(doimiy-boshlanishi)}")
            vaqt = int(doimiy-boshlanishi)
            summary.append(a[0])
            total_points.write(f'summary: {summary}, acc: {summary.count(0) / (len(summary))}, soni: {frames_soni}')

            # if vaqt % 20 == 0:
            #     if summary.count(0) / (len(summary)) > 0:
            #         total_points.write(f'summary: {summary}, acc: {summary.count(0) / (len(summary))}, soni: {frames_soni}')
            #     # summary = []    

        # Release everything after the job is finished
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    else:
        st.write("Error: Unable to open the video file.")
else:
    st.write("Please upload a video file to display.")
