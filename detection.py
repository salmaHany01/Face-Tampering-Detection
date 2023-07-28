from __future__ import division, print_function
from math import ceil
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img

import keras.utils as image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2
from tensorflow.keras.utils import img_to_array
#define the flask app
app = Flask(__name__ , static_folder='static')

#Model saved
IMAGE_MODEL_PATH = "vggface_custom_model_23.h5"
VIDEO_MODEL_PATH = "inception_v3_weights.h5"

#Load your trained models
image_model = load_model(IMAGE_MODEL_PATH)
video_model = load_model(VIDEO_MODEL_PATH)

# function to predict a batch of frames
def batch_predict(frames, model):
    # preprocessing the frames
    imgs = []
    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.true_divide(img, 255)
        imgs.append(img)
    imgs = np.concatenate(imgs, axis=0)

    # make prediction
    preds = model.predict(imgs)
    return preds

# function to predict video with batch processing
def video_predict(video_path, model, batch_size=16):
    # open video file
    cap = cv2.VideoCapture(video_path)

    # get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # process video frames in batches
    results = []
    batch_frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            batch_preds = batch_predict(batch_frames, model)
            results.extend(batch_preds)
            batch_frames = []

    # process any remaining frames in the last batch
    if len(batch_frames) > 0:
        batch_preds = batch_predict(batch_frames, model)
        results.extend(batch_preds)
    
    print("I am hereeeeeeee")

    # aggregate results by second
    seconds = np.array(range(total_frames)) / fps
    seconds = np.floor(seconds).astype(np.int32)
    results = np.array(results)
    
    print(results)
    
    unique_seconds = np.unique(seconds)
    aggregated_results = []
    for s in unique_seconds:
        print("seconds ", s)
        mask = (seconds == s)
        num_frames = np.sum(mask)
        avg_result = np.mean(results[mask])
        aggregated_results.append((s, num_frames, avg_result))

    return aggregated_results

# function to predict image
def image_predict(img_path, model):
    # preprocessing the image
    img = load_img(img_path, target_size=(224,224))
    img = np.true_divide(img, 255)
    img = np.expand_dims(img, axis=0)

    # make prediction
    preds = model.predict(img)
    return preds[0][0]

# function to predict frame
def frame_predict(frame, model):
    # preprocessing the frame
    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # make prediction
    preds = model.predict(img)
    return preds[0][0]

# function to predict file (image or video) with batch processing
def predict_file(file_path, image_model, video_model, batch_size=16):
    # check if the file is an image or video
    file_ext = os.path.splitext(file_path)[1]
    if file_ext.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
        # make prediction for image
        result = image_predict(file_path, image_model)
        if result > 0.5:
            result = "Real"
        else:
            result = "Fake"
    elif file_ext.lower() in ['.mp4', '.avi', '.mov']:
        # process video frames and get results by second
        results = video_predict(file_path, video_model, batch_size=batch_size)
        # format and display results
        is_fake = False
        fake_seconds = []
        for s, num_frames, avg_result in results:
            if avg_result <= 0.5:
                is_fake = True
                fake_seconds.append(s)
        if is_fake:
            result = "Fake </br> Identified at second(s): " + str(fake_seconds)
        else:
            result = "Real"
    else:
        result = "File type not supported."
    return result

# show our home page
@app.route('/', methods=['GET'])
def index():
    return render_template('scanner.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from post request
        f = request.files['file']
        
        # save the file to the uploads folder   
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        result = predict_file(file_path, image_model, video_model, batch_size=16)
        
        return result

if __name__ == 'main':
    app.run(debug=True)