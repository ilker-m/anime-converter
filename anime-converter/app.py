from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask import render_template
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import os
import uuid
import time


app = Flask(__name__)
CORS(app)  # Cross Origin Resource Sharing etkinleştirme

# Geçici dosyaları saklamak için dizin
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')
# Model yükleme - bu örnek için basit bir model
# Gerçek uygulamada daha gelişmiş bir model kullanılabilir
def load_model():
    # Bu fonksiyon gerçek bir modeli yükleyecek
    # Örnek için boş bir model objesi döndürüyoruz
    print("Model yükleniyor...")
    # Burada gerçek bir model yüklenecek
    # model = tf.keras.models.load_model('anime_conversion_model.h5')
    # return model
    return None

# Modeli başlangıçta yükle
model = load_model()

# Basit bir anime dönüşüm fonksiyonu
def convert_to_anime(image):
    """
    Görüntüyü anime tarzına dönüştüren fonksiyon.
    Gerçek bir uygulamada burada ML modeli kullanılacak.
    Bu örnek için basit bir filtre kullanıyoruz.
    """
    # Görüntüyü BGR'den RGB'ye dönüştür
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
        
    # Basit kenar algılama ve yumuşatma
    edge = cv2.Canny(image_rgb, 100, 200)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    
    # Renk doygunluğunu artır
    image_pil = Image.fromarray(image_rgb)
    from PIL import ImageEnhance
    converter = ImageEnhance.Color(image_pil)
    image_pil = converter.enhance(1.5)  # Doygunluğu artır
    
    # Kenarları vurgula
    anime_style = np.array(image_pil)
    anime_style = cv2.addWeighted(anime_style, 0.7, edge, 0.3, 0)
    
    # Daha çizgi film efekti için bilateral filtre
    anime_style = cv2.bilateralFilter(anime_style, 9, 75, 75)
    
    return anime_style

@app.route('/api/convert', methods=['POST'])
def api_convert():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Dosyayı kaydet
    file_id = str(uuid.uuid4())
    timestamp = int(time.time())
    filename = f"{file_id}_{timestamp}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Görüntüyü oku
    img = cv2.imread(file_path)
    if img is None:
        return jsonify({'error': 'Could not read image'}), 400
    
    # Görüntüyü dönüştür
    anime_img = convert_to_anime(img)
    
    # Dönüştürülmüş görüntüyü kaydet
    result_filename = f"anime_{file_id}_{timestamp}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(anime_img, cv2.COLOR_RGB2BGR))
    
    # Sonuç dosyasının URL'sini döndür
    return jsonify({
        'result_url': f"/api/results/{result_filename}",
        'original_url': f"/api/uploads/{filename}"
    })

@app.route('/api/results/<filename>')
def get_result(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

@app.route('/api/uploads/<filename>')
def get_upload(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
