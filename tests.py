import cv2
import numpy as np
import base64

def detect_faces(image):
    # Load the base64 encoded image and convert it to bytes
    image_data = base64.b64decode(image)
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the image with detected faces
    cv2.imshow('Facial Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
#base64_image = "YOUR_BASE64_ENCODED_IMAGE"

#detect_faces(base64_image)