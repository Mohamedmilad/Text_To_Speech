from flask import Flask, render_template, request, jsonify, url_for
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
import torch
import os
import numpy as np
from scipy.io.wavfile import write
from IPython.display import Audio
from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
app = Flask(__name__)

# Load the model and processor once when the server starts
model = SpeechT5ForTextToSpeech.from_pretrained("checkpoint-500")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# Load the speaker embeddings from the saved .pt file
speaker_embeddings = torch.load('speaker_embeddings.pt')

# Ensure the static/audio directory exists
os.makedirs('static/audio', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')  # A simple HTML page with a form and audio player

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form['text']
    
    # Tokenize the input text
    tokenizer = processor.tokenizer
    inputs = tokenizer(text, return_tensors="pt")
    
    # Add speaker embeddings to the inputs
    inputs['speaker_embeddings'] = speaker_embeddings
    
    # Generate speech from the model
    speech = model.generate_speech(**inputs)
    
    with torch.no_grad():
        speech = vocoder(speech)
    
    Audio(speech.numpy(), rate=16000)
    # Convert speech data to 16-bit PCM format
    audio_data = speech.detach().cpu().numpy()
    audio_data = np.clip(audio_data, -1, 1)  # Ensure values are in the range [-1, 1]
    audio_data = (audio_data * 32767).astype(np.int16)  # Scale and convert to int16

    # Save the speech as a WAV file in the static/audio directory
    output_filename = f"static/audio/output_{np.random.randint(1e6)}.wav"
    sample_rate = 16000  # Adjust sample rate based on your model's output
    write(output_filename, sample_rate, audio_data)
    
    # Return the audio file URL for the frontend
    return jsonify({'audio_url': url_for('static', filename=f"audio/{os.path.basename(output_filename)}")})

if __name__ == '__main__':
    app.run(debug=True)
