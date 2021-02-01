from flask import Flask, jsonify ,request, render_template,send_file, make_response
from werkzeug.utils import secure_filename
import pandas as pd
from Api import Api
import sys, os
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

obj = Api()

UPLOAD_FOLDER = 'input/tempData/'
ALLOWED_EXTENSIONS = set(['png','jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    path = request.path
    return render_template('home.html', data=path)

@app.route('/process',methods=['POST'])
def classification_process():
    path = request.path
   
    error = ""
    if 'file' in request.files:
        filetxt = request.files["file"]
        if filetxt and allowed_file(filetxt.filename):
            filename = secure_filename(filetxt.filename)
            print(filename,filetxt.filename)
            filetxt.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            error = "Format file salah"
    
    img = obj.read_img('input/tempData/'+filename)

    imgScale = obj.scale_img(img)
    imgGray = obj.grayscale(imgScale)
    imgHsv = obj.rgb_hsv(imgScale)
    imgThresh = obj.threshold(imgScale)

    obj.save_img(imgScale,'scaledImage.'+filename.split('.')[1])
    obj.save_img(imgGray,'grayscaleImage.'+filename.split('.')[1])
    obj.save_img(imgHsv,'rgbToHsvImage.'+filename.split('.')[1])
    obj.save_img(imgThresh,'tresholdImage.'+filename.split('.')[1])
    
    model = obj.loadModel('data/model_good.h5')
    predictImage = obj.predict(model, 'static/tresholdImage.'+filename.split('.')[1]) 


    try:
        return jsonify({ 'code':200, 'message' : 'Success' ,'data':predictImage}), 200
    except e:
        return jsonify({ 'code':500, 'message' : 'Success', 'error': str(e) }), 500
     
if __name__ == "__main__":
    app.run(debug=True, port=8080)
    # port = int(os.environ.get("PORT",5000))
    # app.run(host="0.0.0.0",port=port)