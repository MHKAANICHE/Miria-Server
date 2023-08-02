from flask import Flask, render_template, request, send_file, make_response
from controllers.Wav_controller import clean_and_save_Wav
import os

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def upload_Wav():
    if request.method == "POST":
        Wav_file = request.files["Wav_file"]
        if Wav_file.filename == "":
            return "No selected file"
        
        cleaned_file_path = clean_and_save_Wav(Wav_file)

        # Read the cleaned file data
        with open(cleaned_file_path, "rb") as cleaned_file:
            cleaned_data = cleaned_file.read()

        response = make_response(cleaned_data)
        response.headers["Content-Disposition"] = f"attachment; filename=cleaned.wav"
        response.headers["Content-Type"] = "audio/wav"

        return response

    return render_template("upload.html")

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    # Open Port https://196.224.153.85:8080/
    app.run(host='0.0.0.0', port=80, debug=True)
    # Local on my pc 
    # app.run(debug=True)
