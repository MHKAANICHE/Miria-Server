import torch
from controllers.Wav_cleaning_model import WavCleaningModel

model = WavCleaningModel()

# Load checkpoint and match specific layers' weights
checkpoint = torch.load('PyTorchModels/model_checkpoint.pth')

# Get the model's state dictionary
model_state_dict = model.state_dict()

# Filter out keys that don't match between checkpoint and model
matched_state_dict = {
    key: value for key, value in checkpoint.items() if key in model_state_dict
}

# Update the model's state dictionary with matched weights
model_state_dict.update(matched_state_dict)

# Load the updated state dictionary into the model
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

def clean_and_save_Wav(Wav_file):
    Wav_data = Wav_file.read()

    # Convert bytes to a PyTorch tensor
    Wav_tensor = torch.tensor(list(Wav_data), dtype=torch.float32)

    # Process Wav_data using your PyTorch model
    with torch.no_grad():
        cleaned_data = model(Wav_tensor.unsqueeze(0))  # Add batch dimension

    # Convert the cleaned tensor back to bytes
    cleaned_bytes = cleaned_data.squeeze().byte().cpu().numpy()

    cleaned_file_path = "temp/cleaned.Wav"
    with open(cleaned_file_path, 'wb') as cleaned_file:
        cleaned_file.write(cleaned_bytes)
    return cleaned_file_path
