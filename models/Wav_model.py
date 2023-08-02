import torch

class WavModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def clean_Wav(self, Wav_data):
        # Process Wav_data using your PyTorch model
        cleaned_data = self.model(torch.tensor(Wav_data))
        return cleaned_data.detach().numpy()  # Return the cleaned data as NumPy array
