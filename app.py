from flask import Flask, render_template, request, send_file, make_response, url_for
from models.Wav_model import WavModel
import os
import librosa
import io
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


# Path to the pre-trained model
model_path = 'D:\portfolio\MiriaData\SoundWebApp\PyTorchModels\model_checkpoint.pth'

app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def upload_Wav():

    image_dir = "static"  # Change this to the actual directory where you want to save the images
    os.makedirs(image_dir, exist_ok=True)

    waveform_image_url = ""  # Placeholder for waveform image URL
    mel_spectrogram_image_url = ""  # Placeholder for Mel spectrogram image URL

    if request.method == "POST":
        Wav_file = request.files["Wav_file"]
        if Wav_file.filename == "":
            return "No selected file"

        # Load the audio file and calculate the waveform and Mel spectrogram
        audio, _ = librosa.load(Wav_file, sr=None)

        # Generate waveform visualization
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=44100)
        waveform_image_path = "static/waveform.png"  # Replace with actual path
        plt.savefig(waveform_image_path)
        plt.close()
        waveform_image_url = waveform_image_path

        # Generate Mel spectrogram visualization
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=44100)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 3))
        librosa.display.specshow(mel_db, sr=44100, x_axis='time', y_axis='mel')
        mel_spectrogram_image_path = "static/mel_spectrogram.png"  # Replace with actual path
        plt.savefig(mel_spectrogram_image_path)
        plt.close()
        mel_spectrogram_image_url = mel_spectrogram_image_path

      
        # Call the clean method on the instance, passing the audio data
        wav_model = WavModel (model_path)
        cleaned_file_path = wav_model.clean(audio)

        # Read the cleaned file data
        with open(cleaned_file_path, "rb") as cleaned_file:
            cleaned_data = cleaned_file.read()

        # Visualize the waveform and Mel spectrogram
        wav_model.visualize_waveform(audio)
        wav_model.visualize_mel_spectrogram(audio)

        response = make_response(cleaned_data)
        response.headers["Content-Disposition"] = f"attachment; filename=cleaned.wav"
        response.headers["Content-Type"] = "audio/wav"

        return response

    return render_template("upload.html", waveform_image_url=url_for('static', filename='waveform.png'), mel_spectrogram_image_url=url_for('static', filename='mel_spectrogram.png'))

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    app.run(debug=True)
