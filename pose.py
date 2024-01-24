from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
import cv2
# import numpy as np
import pandas as pd
import uuid
import joblib

# from ultralytics.utils.ops import scale_coords
# from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

# Load a pretrained YOLOv8m-pose model 
model = YOLO('/home/airi/Makhmud/yolo_pose/Action-Detection-YOLO-Pose-Estimation-Sleep-or-Not-Sleep/models/yolov8n-pose.pt')

# Define column names
column_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# Create an empty DataFrame with the specified column names
df = pd.DataFrame(columns=column_names)

def cut_bbox_on_image(image, x1, y1, x2, y2):

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Cut out the region defined by the bounding box
    cropped_image = image[y1:y2, x1:x2].copy()

    return cropped_image

def draw_bbox_on_image(image, x1, y1, x2, y2):

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

def predict_sleep_or_not_sleep(df):

    # Load the model
    loaded_rf_model = joblib.load('/home/airi/Makhmud/yolo_pose/Action-Detection-YOLO-Pose-Estimation-Sleep-or-Not-Sleep/models/random_forest_model_latest.pkl') 

    # Make predictions on the test set
    pred = loaded_rf_model.predict(df)

    return pred


def predict_pose(image):
    global df
  
    results = model(image)
    # img = cv2.imread(image)

    # Iterate through each result in results
    for result in results:

        boxes = result.boxes

        # Draw keypoints for each person
        person_indx = 0
        for box in boxes:

            xyxy = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            # annotator.box_label(xyxy, model.names[int(box.cls)])
            odam = cut_bbox_on_image(image, xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            # cv2.imwrite(f'/home/airi/Makhmud/yolo_pose/ultralytics/collected_images/{soni}.jpg', odam)

            # Draw person_indx on the image
            full_image = draw_bbox_on_image(image, xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            # cv2.putText(full_image, str(person_indx), (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            person_indx += 1

            # Resize the image to (480, 160)
            odam = cv2.resize(odam, (140, 250))
            predicts = model(odam)

            for predict in predicts:
                
                keypoints_massiv = predict.keypoints.xy
                keypoints_royhat = keypoints_massiv.tolist()

                # Draw keypoints for each person
                for person_keypoint in keypoints_royhat:

                    coordinates = []    
                    people = {}

                    for keypoint_index, key_point in enumerate(person_keypoint):
                        if keypoint_index < 11:

                            # Draw the landmark
                            cv2.circle(odam, (int(key_point[0]), int(key_point[1])), 1, (0, 0, 255), -1)
                            coordinates.append(key_point[0])
                            coordinates.append(key_point[1])
                    people[person_indx] = coordinates

                    # Generate a random file name using uuid
                    random_filename = str(uuid.uuid4())

                    # Check the length of coordinates and compare it with the number of columns in column_names
                    if len(coordinates) == len(column_names):
                        # Transpose the data before appending
                        df = pd.concat([df, pd.DataFrame([coordinates], columns=column_names)], ignore_index=True)

                        sleep_or_not_sleep  = predict_sleep_or_not_sleep(df.tail(len(boxes)))

                    else:
                        print("Number of coordinates does not match the number of columns.")

                    # cv2.imwrite(f'/home/airi/Makhmud/yolo_pose/ultralytics/rasm.jpg', odam)
                    # cv2.imwrite(f'/home/airi/Makhmud/yolo_pose/ultralytics/collected_images/extra/{random_filename}.jpg', odam)
                    # df.to_csv(f'/home/airi/Makhmud/yolo_pose/ultralytics/collected_images/Habib_sleep.csv', index=False)

    return full_image, sleep_or_not_sleep

