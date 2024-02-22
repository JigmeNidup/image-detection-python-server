from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64

from create_image_data import imgToDir
from image_recognize import detectFace, findFace, trainModel

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def hello_world():
	return 'Hello World'

@app.route('/image',methods=['POST'])
def process_image():
    try:
        data = request.json
        imgToDir(data["imgdata"][23:],data["count"],data["username"])
        trainModel()
        return {"status":"true"}
    except:
        return {"status:":"false"}

@app.route('/detect',methods=['POST'])
def detect_image():
    try:
        data = request.json
        result = detectFace(data["imgdata"][23:],data["username"])
        return {"status":"true","result":result}
    except:
        return {"status:":"false"}

@app.route('/find',methods=['POST'])
def find_image():
    try:
        data = request.json
        result = findFace(data["imgdata"][23:])
        return {"status":"true","result":result}
    except:
        return {"status:":"false"}

@app.route('/train',methods=['GET'])
def train_image():
    try:
        trainModel()
        return {"status":"true",}
    except:
        return {"status:":"false"}


if __name__ == '__main__':
	app.run()
