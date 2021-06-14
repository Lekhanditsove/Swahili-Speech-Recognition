import os

from swahili_speech_recognition_service import Swahili_Speech_Recognition_Service

from flask import Flask, render_template, request

from logging import getLogger

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio', methods=['POST'])
def audio():
    with open('tmp/audio.wav', 'wb') as f:
        f.write(request.data)
        
    # instantiate Swahili Speech Recognition service singleton and get prediction
    print("Check 1")
    kss = Swahili_Speech_Recognition_Service()

    print("Check 2")
    predicted_transcription = kss.predict('tmp/audio.wav')

    print("Check 3")
    # we don't need the audio file any more - let's delete it!
    os.remove('tmp/audio.wav')

    print("Check 4")
    result = str(predicted_transcription)

    print("Check 5")
    return result

if __name__ == "__main__":
    app.logger = getLogger('audio-gui')
    app.run(debug=True)