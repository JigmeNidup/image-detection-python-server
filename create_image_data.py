import cv2, sys, numpy, os
import base64

haar_file = 'haarcascade_frontalface_default.xml'

datasets = 'datasets'

(width, height) = (130, 100)	

def imgToDir(img_raw,count,username):
    sub_data = username

    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
	    os.mkdir(path)
       # Load the base64 encoded image and convert it to bytes
    image_data = base64.b64decode(img_raw)
    nparr = numpy.frombuffer(image_data, numpy.uint8)
    # Decode the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    print("done")
