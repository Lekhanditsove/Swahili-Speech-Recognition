from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa as lb
import torch
import timeit

class _Swahili_Speech_Recognition_Service:

    _instance = None
    _processor = None
    _model = None

    def predict(self, file_path):

        # Initialize the tokenizer
        processor = self._processor

        # Initialize the model
        model = self._model

        # Read the sound file
        waveform, rate = lb.load(file_path, sr=16000)

        # Tokenize the waveform
        input_values = processor(waveform, return_tensors='pt').input_values

        # Retrieve logits from the model
        logits = model(input_values).logits

        # Take argmax value and decode into transcription
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.batch_decode(predicted_ids)

        return transcription

def Swahili_Speech_Recognition_Service():
    """Factory function for _Swahili_Speech_Recognition_Service class.

    :return _Swahili_Speech_Recognition._instance (_Swahili_Speech_Recognition Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if  _Swahili_Speech_Recognition_Service._instance is None:

        _Swahili_Speech_Recognition_Service._instance = _Swahili_Speech_Recognition_Service()

        # Initialize the tokenizer
        _Swahili_Speech_Recognition_Service._processor = Wav2Vec2Processor.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw")

        # Initialize the model
        _Swahili_Speech_Recognition_Service._model = Wav2Vec2ForCTC.from_pretrained("alokmatta/wav2vec2-large-xlsr-53-sw")

    return _Swahili_Speech_Recognition_Service._instance


if __name__ == "__main__":
    start = timeit.default_timer()
    # create 2 instances of the Swahili Speech Recognition
    kss = Swahili_Speech_Recognition_Service()
    kss1 = Swahili_Speech_Recognition_Service()

    # check that different instances of the Swahili Speech Recognition service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    predicted_transcription = kss.predict("tmp/sample.wav")

    # Print the output
    print(predicted_transcription)

    # All the program statements
    stop = timeit.default_timer()
    execution_time = stop - start

    print("Program Executed in " + str(execution_time))  # It returns time in seconds
