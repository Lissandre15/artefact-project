import os
import uuid
from PIL import Image
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
from keras.models import load_model
import cloudpickle
from functions.preprocessing import load_and_convert_image, remove_bg
from functions.yolo import yolo_image

app = Flask(__name__)

# Define upload folders
UPLOAD_FOLDER_img_cars = 'uploads/original_img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_img_cars

UPLOAD_FOLDER_img_rmbg = 'uploads/img_rmbg'
app.config['UPLOAD_FOLDER_rmbg'] = UPLOAD_FOLDER_img_rmbg

UPLOAD_FOLDER_img_cars_array = 'uploads/img_array'
app.config['UPLOAD_FOLDER_ARRAY'] = UPLOAD_FOLDER_img_cars_array

UPLOAD_FOLDER_ID = 'uploads/img_id'
app.config['UPLOAD_FOLDER_id'] = UPLOAD_FOLDER_ID

UPLOAD_FOLDER_img_yolo = 'uploads/img_yolo'
app.config['UPLOAD_FOLDER_img_yolo'] = UPLOAD_FOLDER_img_yolo

# Function to create folders
def create_upload_folders():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_rmbg'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_ARRAY'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_id'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_img_yolo'], exist_ok=True)

# Call the function
create_upload_folders()

# Load  VGG model and XGBoost model
vgg_model = load_model("models/vgg.h5")
xgb_model = cloudpickle.load(open("models/xgboost.pkl", 'rb'))

# Route for home page
@app.route('/', methods=['GET'])
def price_car_prediction_index(): 
    return render_template('base.html')

# Route to retrieve YOLO image
@app.route('/upload/img_yolo/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_img_yolo'], filename)

# Route to handle form input and price prediction
@app.route('/predict/', methods=['POST'])
def input_client():
    if request.method == 'POST':
        # Check if an image is uploaded
        if 'image' not in request.files or request.files['image'].filename == '':
            return "No image selected", 400

        # Save uploaded image
        file = request.files['image']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img = Image.open(file_path)
        print(f'Image uploaded and saved as {file.filename}', 200)

        # Apply YOLO
        yolo_img = yolo_image(img)

        # Remove background
        img_rmbg = remove_bg(file_path)
        file_path_rmbg = os.path.join(app.config['UPLOAD_FOLDER_rmbg'], file.filename)
        img_rmbg.convert("RGB").save(file_path_rmbg)

        # Convert image to array
        img_array = load_and_convert_image(file_path_rmbg, target_size=(224, 224))
        file_path_array = os.path.join(app.config['UPLOAD_FOLDER_ARRAY'], file.filename)
        np.save(file_path_array, img_array)
        print(f"Image converted to array: {file.filename}", 200)

        # Preprocess image for the model
        img_array_scaled = img_array / 255
        img_array_scaled = np.expand_dims(img_array_scaled, axis=0)

        # Predict car condition using VGG model
        condition = np.argmax(vgg_model.predict(img_array_scaled))
        print(f'Condition: {condition}')

        # Prepare input data for price prediction
        inputs = {
            'year': float(request.form["year"]),
            'new_condition': float(condition), 
            'odometer': float(request.form["odometer"]),
            'make': request.form["make"],
            'model': request.form["model"],
            'body': request.form["body"],
            'transmission': request.form["transmission"],
            'state': request.form["state"],
            'color': request.form["color"],
            'interior': request.form["interior"]
            }
        input_df = pd.DataFrame([inputs])

        # Predict car price using XGBoost model
        pred = xgb_model.predict(input_df)
        print('Prediction completed')

        # Save YOLO image and unique ID
        img_yolo_path = os.path.join(app.config['UPLOAD_FOLDER_img_yolo'], file.filename)
        yolo_img.save(img_yolo_path)
        
        # Generate unique ID and save information
        myuuid = uuid.uuid4()
        name_of_file = f'{file.filename} -id: {myuuid}'
        file_path_id = os.path.join(app.config['UPLOAD_FOLDER_id'], name_of_file)
        id_history = {
            "id": name_of_file,
            "image": file.filename,
            "img_array": img_array,
            "condition": condition,
            "price_predict": pred[0]
            }
        
        # Save data to a pickle file
        with open(f'{file_path_id}.pkl', 'wb') as f:
            cloudpickle.dump(id_history, f)

        print(f'Data saved as {file_path_id}.pkl')
        return render_template('prediction.html', price=int(pred[0] * 0.8), img_filename=file.filename)

# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)