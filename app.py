from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
Flatten, Dense, Activation, Dropout,LeakyReLU
from PIL import Image
from fungsi import make_model


# ====== FLASK SETUP ======

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

model = None

NUM_CLASSES = 12
class_folder = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']


app   = Flask(__name__, static_url_path='/static')

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def index():
    return render_template('index.html')

# [Routing untuk Halaman About]
@app.route("/about")
def about():
    return render_template('about.html')

# [Routing untuk Halaman team]
@app.route("/team")
def team():
    return render_template('team.html')

# [Routing untuk Halaman apikasi]
@app.route("/aplikasi", methods=['GET', 'POST'])
def aplikasi():
    return render_template('aplikasi.html')


# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image         = Image.open('.' + gambar_prediksi)
			
			# Mengubah Ukuran Gambar
			test_image_resized = test_image.resize((32, 32))
			
			# Konversi Gambar ke Array
			image_array        = np.array(test_image_resized)
			test_image_x       = (image_array / 255) - 0.5
			test_image_x       = np.array([image_array])
			
			# Prediksi Gambar
			y_pred_test_single         = model.predict(test_image_x)
			y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)
			
			hasil_prediksi = class_folder[y_pred_test_classes_single[0]]
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
        

# =[Main]========================================		

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = make_model()
	model.load_weights("garbage_classification_model.h5")

	# Run Flask di Google Colab menggunakan ngrok
	app.run()
