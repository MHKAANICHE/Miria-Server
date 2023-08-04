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

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

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


# def generate_mel_spectrogram(audio_path):
#     waveform, _ = torchaudio.load(audio_path)
#     mel_transform = T.MelSpectrogram(n_mels=64, n_fft=400, hop_length=160)
#     mel_spectrogram = mel_transform(waveform)
    
#     plt.figure(figsize=(10, 4))
#     plt.imshow(mel_spectrogram[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
#     plt.title('Mel Spectrogram Visualization')
#     plt.xlabel('Frame')
#     plt.ylabel('Mel Filter')
#     plt.tight_layout()

#     image_stream = BytesIO()
#     plt.savefig(image_stream, format='png')
#     plt.close()
#     image_stream.seek(0)
#     return image_stream


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/waveform/<filename>')
def waveform(filename):
    waveform_image_stream = generate_waveform_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_file(waveform_image_stream, mimetype='image/png')

# @app.route('/mel_spectrogram/<filename>')
# def mel_spectrogram(filename):
#     mel_spectrogram_image_stream = generate_mel_spectrogram(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#     return send_file(mel_spectrogram_image_stream, mimetype='image/png')

# Function to apply the audio modification
def apply_audio_modification(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Reverse the waveform
    modified_waveform = waveform.flip(dims=[1])
    
    return modified_waveform

@app.route('/compare', methods=['POST'])
def compare():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        original_waveform_image_stream = generate_waveform_image(file_path)
        # original_mel_spectrogram_image_stream = generate_mel_spectrogram(file_path)
        
        modified_waveform = apply_audio_modification(file_path)
        modified_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_' + file.filename)
        torchaudio.save(modified_audio_path, modified_waveform, 44100)  # Adjust sample rate if needed
        
        modified_waveform_image_stream = generate_waveform_image(modified_audio_path)
        # modified_mel_spectrogram_image_stream = generate_mel_spectrogram(modified_audio_path)
        
        return render_template(
            'index.html',
            original_audio_url=url_for('uploaded_file', filename=file.filename),
            modified_audio_url=url_for('uploaded_file', filename='modified_' + file.filename),
            original_waveform_url=url_for('waveform', filename=file.filename),
            # original_mel_spectrogram_url=url_for('mel_spectrogram', filename=file.filename),
            modified_waveform_url=url_for('waveform', filename='modified_' + file.filename)
            # modified_mel_spectrogram_url=url_for('mel_spectrogram', filename='modified_' + file.filename),
        )
    
    return redirect(request.url)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()

