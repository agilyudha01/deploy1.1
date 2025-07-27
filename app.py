import time
import psutil
import os
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import torch
from donut import DonutModel  # pastikan sudah install dan tersedia
# inisialisasi Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Start timer
        start_time = time.time()

        # Cek memori sebelum proses
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss  # in bytes

        # Ambil string base64 dari JSON field "image"
        im_b64 = request.json['image']

        # Decode base64 ke bytes
        im_bytes = base64.b64decode(im_b64)

        # Ubah jadi objek file
        im_file = BytesIO(im_bytes)

        # Buka dengan PIL
        img = Image.open(im_file).convert("RGB")

        # Load model
        model = DonutModel.from_pretrained("model")
        model.eval()

        task_prompt = "<s_dataset_name_augmented>"

        with torch.no_grad():
            output = model.inference(image=img, prompt=task_prompt)

        # Hitung durasi dan memori sesudah
        duration = time.time() - start_time
        mem_after = process.memory_info().rss
        mem_used_mb = (mem_after - mem_before) / (1024 * 1024)  # in MB

        print(f"[INFO] Request took {duration:.4f} seconds")
        print(f"[INFO] Memory used: {mem_used_mb:.2f} MB")

        return jsonify({
            "output": output,
            "resource": {
                "time_seconds": round(duration, 4),
                "memory_used_mb": round(mem_used_mb, 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400    



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
