import os
from flask import Flask, request, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from lime import lime_image
from skimage.segmentation import mark_boundaries


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL = tensorflow.keras.models.load_model('saved_model')
EXPLAINER = lime_image.LimeImageExplainer()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    params = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pred_filename = f'pred_{filename}'
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            prediction = predict_pneumonia(img_path)
            params['prediction'] = prediction
            show_predictions(img_path, pred_filename)
            params['img_url'] = os.path.join(app.config['UPLOAD_FOLDER'], pred_filename)
    return render_template('upload.html', params=params)


@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def process_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict_pneumonia(img_path):
    img = process_img(img_path)
    prediction = MODEL.predict(img)[0, 0]
    if prediction >= 0.5:
        return 'Healthy', round(prediction*100, 2)
    else:
        return 'Sick', round((1 - prediction)*100, 2)


@app.route('/explain', methods=['POST'])
def show_predictions(img_path, pred_filename):
    # Calculating explanation
    explanation = EXPLAINER.explain_instance(process_img(img_path)[0].astype('double'),
                                             MODEL.predict,
                                             top_labels=2,
                                             hide_color=0,
                                             num_samples=1000,
                                             distance_metric='cosine')
    temp, mask = explanation.get_image_and_mask(label=0,
                                                positive_only=False,
                                                num_features=15,
                                                hide_rest=False,
                                                min_weight=0.0000004)
    tempp = np.interp(temp, (temp.min(), temp.max()), (0, +1))
    # Saving resulted image
    plt.imshow(cv2.imread(img_path))
    plt.imshow(mark_boundaries(tempp, mask))
    plt.axis('off')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], pred_filename))


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run(host='0.0.0.0')