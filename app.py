from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import torch
import torchaudio
import torchaudio.transforms as T
from PIL import Image, ImageDraw
import soundfile as sf  # For saving audio files
from your_model import NoisingAutoencoder  # Import your model class
import numpy as np
import time

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_waveform_image(audio_path):
    waveform, _ = librosa.load(audio_path, sr=None)
    image = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw waveform
    width, height = image.size
    for i in range(len(waveform) - 1):
        x0 = int(i / len(waveform) * width)
        x1 = int((i + 1) / len(waveform) * width)
        y0 = int((waveform[i] + 1) / 2 * height)
        y1 = int((waveform[i + 1] + 1) / 2 * height)
        draw.line((x0, y0, x1, y1), fill='black')
    
    image_stream = BytesIO()
    image.save(image_stream, format='png')
    image_stream.seek(0)
    return image_stream

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define your model instance

# First, initialize the model
model = NoisingAutoencoder()

# Load the weights from the saved state_dict
model.load_state_dict(torch.load("5ouza3bol_hamdi.pt"))

# Set the model to evaluation mode
model.eval()

def apply_audio_modification(audio_path, filename):
    start_time = time.time()
    # Get the absolute path of the audio file
    audio_path = os.path.join(audio_path)
    # print ("Debug 1 : - audio_path",audio_path)

    # Load your new audio file using librosa
    new_audio, sr = librosa.load(audio_path, sr=None)

    # Normalize audio to between -1 and 1
    audio = new_audio / np.abs(new_audio).max()

    # Reshape the audio data to add an extra dimension for the model
    # The shape should be (1, 1, length_of_audio)
    audio = audio[np.newaxis, np.newaxis, :]

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float()

    # Pass the audio through the model
    with torch.no_grad():
        noisy_audio = model(audio_tensor)

    # The output will be a tensor. You can convert it to a numpy array like this:
    noisy_audio = noisy_audio.numpy()

    # The shape of the output will be (1, 1, length_of_audio). 
    # If you want to convert it back to a 1D array, you can do this:
    noisy_audio = np.squeeze(noisy_audio)
    
    # Save the modified waveform as a new audio file
    modified_audio_name = os.path.splitext(os.path.basename(audio_path))[0] + '_modified.wav'

    modified_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_' + filename)

    sf.write(modified_audio_path, noisy_audio, sr)  # Assuming mono audio
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"apply_audio_modification - Function execution time: {execution_time:.4f} seconds")
    return modified_audio_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # print ("Debug 0 : - audio_path ",os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/waveform/<filename>')
def waveform(filename):
    start_time = time.time()
    waveform_image_stream = generate_waveform_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"waveform - Function execution time: {execution_time:.4f} seconds")
    return send_file(waveform_image_stream, mimetype='image/png')

@app.route('/compare', methods=['POST'])
def compare():
    start_time = time.time()
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # print ("Debug 3 : - audio_path ", os.path.exists(app.config['UPLOAD_FOLDER']) )

    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        # print ("Debug 4 : - audio_path ", os.path.join(app.config['UPLOAD_FOLDER'], file.filename) )
        file.save(file_path)

        # Original
        original_waveform_image_stream = generate_waveform_image(file_path)
        # Modification
        modified_audio_path = apply_audio_modification(file_path, file.filename)
        modified_waveform_image_stream = generate_waveform_image(file_path)
        # modified_mel_spectrogram_image_stream = generate_mel_spectrogram(modified_audio_path)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Compare - Function execution time: {execution_time:.4f} seconds")
        return render_template(
            'index.html',
            original_audio_url=url_for('uploaded_file', filename=file.filename),
            modified_audio_url=url_for('uploaded_file', filename='modified_' + file.filename),
            original_waveform_url=url_for('waveform', filename=file.filename),
            # original_mel_spectrogram_url=url_for('mel_spectrogram', filename=file.filename),
            modified_waveform_url=url_for('waveform', filename='modified_' + file.filename),
            # modified_mel_spectrogram_url=url_for('mel_spectrogram', filename='modified_' + file.filename),
            
        )
    

    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
    # app.run()

