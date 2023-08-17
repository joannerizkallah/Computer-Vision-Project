from flask import Flask, redirect, url_for, request, render_template, session, jsonify
import json
import cv2
import os
import onnxruntime
import torch
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = '/Uploads'
#Define allowed files
#LLOWED_EXTENSIONS = {'png', 'jpg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'Apparently I should have a key to make this work'
img_filename = ""
#WORKING
def print_labels():
    f = open('./json/objectclasses.json', "rb")
    data = json.load(f)
    str = []
    for l in data:
        str.append("Label " + "% s" % l["Id"] +": " + "% s" % l["Name"] )
    f.close()
    return "\n".join(str)

#WOKRING
@app.route('/')
def main():
    return render_template('models.html')

#WORKING
@app.route('/models', methods = ['POST','GET'])
def list_models():
    model_files = [ml for ml in os.listdir('./models/')]
    models =[]
    for model in model_files:
        models.append(model)
    if request.method == 'POST':
        if request.form.get('action1') == 'yolov8s':
            session['model_selected'] = 'yolov8s'
            return redirect(url_for('run', model = 'yolov8s'))
        elif  request.form.get('action2') == 'yolov5s':
            session['model_selected'] = 'yolov5s'
            return redirect(url_for('run', model = "yolov5s"))
        else:
            return "hi"

@app.route('/run/<model>')    
def index2(model):
    return render_template('labels.html', selected_model = model)

@app.route('/run/<model>', methods = ['POST','GET'])
def run(model):
    if request.method == 'POST':
        if request.form.get('action1') == 'List labels':
            return redirect(url_for('list_labels',models = model)) # do something
        elif  request.form.get('action2') == 'Test Model':
            return redirect('/run/<model>/test') # do something else
        else:
            return "hi"

@app.route('/<models>/list')
def list_labels(models):
 labels = []
 if models == "yolov5s":
    labels = print_labels()
    html = """\
    <html>
      <head></head>
        <body>
        <form action = "http://localhost:5000/run/list">
        <h1>{code}</h1>
        </form>
        </body>
    </html>
    """.format(code=labels)
    return html
 else:
    labels = print_labels()
    html = """
    <html>
      <head></head>
        <body>
        <form action = "http://localhost:5000/run/list">
        <h1>{code}</h1>
        </form>
        </body>
    </html>
    """.format(code=labels)
    return "same labels as yolov5s, check Inference_API.py for explanation"
# note that if the models had different labels, I'd pass in models to the print_labels function and list accordingly. That's hwy I created two separate functions


#WORKING
@app.route('/run/<model>/test', methods=['POST', 'GET'])
def index(model):
    return render_template('upload_image.html')
 
 #WORKING
@app.route('/<model>/test', methods= ['POST'])
def uploadFile(model):
    if request.method == 'POST':
        if request.form.get('submit') == 'Submit':
            # Upload file flask
            uploaded_img = request.files['uploaded-file']
            # Extracting uploaded data file name
            img_filename = secure_filename(uploaded_img.filename)
            # Upload file to database (defined uploaded folder in static path)
            uploaded_img.save("./Uploads/" + img_filename)
            # Storing uploaded file path in flask session
            session['uploaded_img_file_path'] = img_filename
            onnx_model = session['model_selected'] + ".onnx"
            #yolo_m = torch.hub.load('./models/', onnx_model,source='local')
            
            ort_sess = onnxruntime.InferenceSession("./models/" + onnx_model, None)
            img = cv2.imread("./Uploads/" + img_filename, cv2.IMREAD_UNCHANGED)  # PIL image
            resized = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA)
            img = Image.open('./Uploads/' + img_filename).convert("RGB")
            img = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
            img = np.ascontiguousarray(img)
            np_image = torch.from_numpy(img)
            np_image = np.expand_dims(np_image, axis=0)
            np_image = np_image.astype(np.float32)
            resized = np.expand_dims(resized, axis=0).astype(np.float32)
            ort_inputs = {ort_sess.get_inputs()[0].name: np_image}
            ort_outs = ort_sess.run(None, ort_inputs)

            from yolov5.utils.general import non_max_suppression, xyxy2xywh

            output= torch.from_numpy(np.asarray(ort_outs))
            out = non_max_suppression(output, conf_thres=0.2, iou_thres=0.5)[0]
            xyxy = out[:,:4]
            xywh = xyxy2xywh(xyxy)
            out[:, :4] = xywh
            num_xywh= xywh.numpy().tolist()
            xywh_dict = convert(num_xywh[0])
            return jsonify(xywh_dict)
    return "Kherbet ldene..."
def convert(lst):
   res_dict = {}
   res_dict['X'] = lst[0]
   res_dict['Y'] = lst[1]
   res_dict['W'] = lst[2]
   res_dict['H'] = lst[3]
   return res_dict
#@app.route('/show_image')
#def displayImage():
#    # Retrieving uploaded file path from session
#    img_file_path = session.get('uploaded_img_file_path', None)
    #run it through model
    # Display image in Flask application web page
#    return render_template('show_image.html', user_image = img_file_path)
 
@app.route("/<model>/detect", methods = ['GET','POST'])
def test(model):
    onnx_model = session['model_selected'] + ".onnx"
    ort_sess = onnxruntime.InferenceSession("./models/" + onnx_model, None)
    # get the name of the first input of the model
    outputs = ort_sess.run(None, {'input': "./Uploads/"+img_filename})

    return "hi"

if __name__ == '__main__':
   app.run(debug = True)