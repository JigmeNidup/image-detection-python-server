# It helps in identifying the faces
import cv2, sys, numpy, os
import base64
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

(width, height) = (130, 100)

(images, labels, names, id) = ([], [], {}, 0)
model = cv2.face.LBPHFaceRecognizer_create()

def trainModel():
    print("pt")
    global images, labels, names, id, model
    (images, labels, names, id) = ([], [], {}, 0)
    
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)

# trainModel()
            
def findFace(img_raw):
    image_data = base64.b64decode(img_raw)
    nparr = numpy.frombuffer(image_data, numpy.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        if prediction[1]<90:
            return {"result":True,"data":names[prediction[0]]}
        else:
            return {"result":False}
      