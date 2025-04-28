
########################################################################tTraining######################################################################################################################################################

from ultralytics import YOLO
model=YOLO('yolo11x.pt')
model.train(data=r'C:\Users\Z044837\Python\ai\Images\data.yaml', batch=2, imgsz=640, epochs=100, workers=1)

#######################################################################testing#############################################################################################################
# Performing inference on trained dataset
import cv2
from matplotlib import pyplot as plt
import easyocr
import os
from PIL import Image
from ultralytics import YOLO
import os
import easyocr
import shutil

# Load the trained model
trained_model_path = r'C:\Users\Z044837\Python\ai\Images\column\column.pt'
model = YOLO(trained_model_path)

# List of test images 
test_images = [r'C:\Users\Z044837\Python\ai\Images\Annot_test\1.png', r'C:\Users\Z044837\Python\ai\Images\Annot_test\2.png',r'C:\Users\Z044837\Python\ai\Images\Annot_test\3.png',r'C:\Users\Z044837\Python\ai\Images\Annot_test\4.png']

# Perform inference and display results
for img_path in test_images:
    results = model.predict(source=img_path, conf=0.3,save_crop=True, project=r'C:\Users\Z044837\Python\ai\Images\column', name='inference', exist_ok=True )  # Adjust confidence threshold if needed
#extraction of number and image

import easyocr
import matplotlib.pyplot as plt
import cv2

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en', 'hi'])  # 'hi' for Hindi, 'en' for English

# Load your image
image_path = 'path_to_your_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# OCR detection
results = reader.readtext(image_path)

# Display results
for (bbox, text, confidence) in results:
    print(f"Detected text: {text} (Confidence: {confidence:.2f})")
    
    # Draw bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Show the output image
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
