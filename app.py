import os
import shutil
from flask import Flask, request, jsonify, url_for, Response
import cv2
import numpy as np
from rmn import RMN
from werkzeug.utils import secure_filename
from flask_cors import CORS
from collections import Counter
import time


app = Flask(__name__)
CORS(app)
m = RMN()

OUTPUT_DIR = os.path.join("static", "output_images")
TEXT_DIR = os.path.join("static", "text_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


def clear_output_files():
    for directory in [OUTPUT_DIR, TEXT_DIR]:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def save_detected_emotion(filename, emo_label, image):
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{emo_label}{ext}"
    output_path = os.path.join(OUTPUT_DIR, new_filename)
    cv2.imwrite(output_path, image)
    return new_filename, output_path


@app.route('/clear_images', methods=['POST'])
def clear_images():
    clear_output_files()
    return jsonify({"message": "Files cleared successfully"}), 200


@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No image files provided"}), 400

    results = []
    emotion_records = []
    total_files = len(files)
    errors = []

    for idx, file in enumerate(files):
        filename = secure_filename(file.filename)
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            errors.append({"filename": filename, "error": "Unable to decode the image file"})
            continue

        result = m.detect_emotion_for_single_frame(image)
        if not result:
            emo_label = "nofacialexpressions"
            image = cv2.putText(image, emo_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            new_filename, _ = save_detected_emotion(filename, emo_label, image)
            emotion_records.append(f"{filename}: {emo_label}\n")
            image_url = url_for('static', filename=f'output_images/{new_filename}', _external=True)
            results.append({
                "filename": new_filename,
                "emotions": [{"emo_label": emo_label}],
                "url": image_url
            })
            errors.append({"filename": filename, "error": "No faces detected in the image"})
            continue

        image = m.draw(image, result)
        emo_label = result[0]['emo_label']
        new_filename, _ = save_detected_emotion(filename, emo_label, image)

        emotion_records.append(f"{filename}: {emo_label}\n")
        image_url = url_for('static', filename=f'output_images/{new_filename}', _external=True)

        results.append({
            "filename": new_filename,
            "emotions": result,
            "url": image_url
        })

    text_file_path = os.path.join(TEXT_DIR, "detected_emotions.txt")
    with open(text_file_path, "w") as text_file:
        text_file.writelines(emotion_records)

    response = {
        "results": results,
        "errors": errors
    }

    return jsonify(response)


def event_stream():
    while True:
        for progress in range(0, 101, 10):
            yield f"data: {progress}\n\n"
            time.sleep(1)


@app.route('/progress')
def progress():
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/get_image_count', methods=['GET'])
def get_image_count():
    text_file_path = os.path.join(TEXT_DIR, "detected_emotions.txt")
    try:
        with open(text_file_path, "r") as text_file:
            lines = text_file.readlines()
        image_count = len(lines)
        return jsonify({"image_count": image_count}), 200
    except FileNotFoundError:
        return jsonify({"image_count": 0}), 200


@app.route('/get_emotion_counts', methods=['GET'])
def get_emotion_counts():
    text_file_path = os.path.join(TEXT_DIR, "detected_emotions.txt")
    emotion_counter = Counter()
    try:
        with open(text_file_path, "r") as text_file:
            lines = text_file.readlines()
            for line in lines:
                emotion_label = line.strip().split(": ")[1].strip()
                emotion_counter[emotion_label] += 1

        return jsonify(dict(emotion_counter)), 200
    except FileNotFoundError:
        return jsonify({}), 200


@app.route('/get_processed_images', methods=['GET'])
def get_processed_images():
    text_file_path = os.path.join(TEXT_DIR, "detected_emotions.txt")
    images = []
    try:
        with open(text_file_path, "r") as text_file:
            lines = text_file.readlines()
            for line in lines:
                filename, emo_label = line.strip().split(": ")
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{emo_label}{ext}"
                image_url = url_for('static', filename=f'output_images/{new_filename}', _external=True)
                images.append({
                    "filename": new_filename,
                    "url": image_url,
                    "emo_label": emo_label
                })
        return jsonify(images), 200
    except FileNotFoundError:
        return jsonify([]), 200


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)


""" Facial Recognition Detection API Credits goes to Pham Luan, The Huynh Vu, and Tuan Anh Tran. "Facial Expression Recognition using Residual Masking Network". In: Proc. ICPR. 2020. @inproceedings{pham2021facial, title={Facial expression recognition using residual masking network}, author={Pham, Luan and Vu, The Huynh and Tran, Tuan Anh}, booktitle={2020 25th International Conference on Pattern Recognition (ICPR)}, pages={4513--4519}, year={2021}, organization={IEEE} } """