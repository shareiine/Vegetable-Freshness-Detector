import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import time

# load model and assign to each class to be identified
model = load_model("vegetable_detector.h5")
lb = LabelBinarizer()
lb.classes_ = ['cabbage_fresh', 'carrot_rotten', 'cabbage_rotten', 'carrot_fresh', 'potato_fresh', 'potato_rotten']

# load nutritional data
nutritional_values = pd.DataFrame({
    "name": ["carrot", "cabbage", "potato"],
    "calories": [41, 25, 77],
    "protein_g": [0.9, 1.3, 2.0],
    "carbohydrates_g": [9.6, 5.8, 17.5],
    "fiber_g": [2.8, 2.5, 2.2]
})

# determine prediction threshold for unknown vegetable
PREDICTION_ACCURACY = 0.7

# predict each frame
def predict_frame(frame):
    image = cv2.resize(frame, (32, 32))
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    label_index = np.argmax(prediction)
    label = lb.classes_[label_index]
    confidence = prediction[label_index]

    vegetable, freshness = label.split("_")
    
    return vegetable.capitalize(), freshness.capitalize(), confidence, label, prediction

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    flipped_frame = cv2.flip(frame, 1)
    display_frame = cv2.resize(flipped_frame, (640, 480))
    
    # detect vegetable and determine freshness
    vegetable, freshness, confidence, raw_label, prediction = predict_frame(flipped_frame)

    # check if vegetable (carrot, cabbage, potato) is detected
    if confidence >= PREDICTION_ACCURACY:
        # define bounding box area (centered, arbitrary size)
        box_width, box_height = 150, 150
        center_x, center_y = display_frame.shape[1] // 2, display_frame.shape[0] // 2
        top_left = (center_x - box_width // 2, center_y - box_height // 2)
        bottom_right = (center_x + box_width // 2, center_y + box_height // 2)

        # draw bounding box
        cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 255), 2)
        # if the predicted vegetable is a fresh carrot with low confidence, it is potato due to similar appearance
        if vegetable.lower() == "carrot" and freshness == "Fresh" and confidence < 0.85:
            freshness = "Fresh"
            label_text = f"Potato - {freshness} ({confidence*100:.2f}%)"
        else:
            label_text = f"{vegetable} - {freshness} ({confidence*100:.2f}%)"
        nutritional_info = ""
        if freshness == "Fresh":
            vegetable_data = nutritional_values[nutritional_values['name'].str.lower() == vegetable.lower()]
            if not vegetable_data.empty:
                nutritional_info = " | ".join(
                    f"{nutrient}: {value}" for nutrient, value in vegetable_data.iloc[0].items() if nutrient != "name"
                )
            else:
                nutritional_info = "Nutritional data not available"
        else:
            nutritional_info = "Rotten vegetable detected"
    else: # not among vegetables
        label_text = f"Uncertain detection (Confidence: {confidence*100:.2f}%)"
        nutritional_info = "Unable to classify the vegetable accurately"

    # display vegertable label and freshness
    cv2.putText(display_frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, nutritional_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Vegetable Detector", display_frame)
    time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
