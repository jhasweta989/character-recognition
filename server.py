import licence_detection
import os
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__,template_folder='template', static_folder='static')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        #preds = ocr.get_text(file_path)
        preds= licence_detection.get_Licence_no(file_path)
        return preds
    return None


if __name__ == "__main__":
    print("Starting Python Flask Server For getting text...")
    app.run(host="127.0.0.1", port=5000)