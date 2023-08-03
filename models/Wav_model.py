import torch
import torchaudio
import io
import wave
import os
import torch
import librosa
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class WavModel:
    def __init__(self, model_path):
        try:
            self.model = torch.load(model_path)
            self.model.eval()
        except Exception as e:
            try: 
                self.model = torch.load_state_dict(model_path)
                print("Model loaded as OrderedDict.")
            except Exception as e:
                print(f"An error occurred while loading the model: {e}")
                self.model = None

    def clean(self, wav_data):
        if self.model is None:
            print("Model was not loaded successfully. Cannot proceed with cleaning.")
            return None
        
        audio_tensor, _ = torchaudio.load(io.BytesIO(wav_data))
        cleaned_data = self.model(audio_tensor)
        cleaned_bytes = torchaudio.save(io.BytesIO(), cleaned_data, sample_rate=audio_tensor.shape[1], format="wav")
        cleaned_file_path = "temp/cleaned.wav"
        with open(cleaned_file_path, 'wb') as cleaned_file:
            cleaned_file.write(cleaned_bytes.getbuffer())
        return cleaned_file_path
    
    def load_audio_data(self, audio_bytes):
        with io.BytesIO(audio_bytes) as f:
            audio = wave.open(f, 'rb')
            frames = audio.readframes(audio.getnframes())
        return frames
    
    def load_wav_files(self, data_dir):
        audio_data = []
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            audio_data.append(audio)
        return audio_data

    def visualize_waveform(self, audio_data):
        plt.figure(figsize=(10,3))
        librosa.display.waveshow(audio_data, sr=44100)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def visualize_mel_spectrogram(self, audio_data):
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=44100)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10,3))
        librosa.display.specshow(mel_db, sr=44100, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency")
        plt.tight_layout()
        plt.show()
