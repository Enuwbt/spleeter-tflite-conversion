import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from scipy.io.wavfile import write as wav_write

current = Path(__file__).resolve().parent
parent = current.parent
model_path = parent / "tflites" / "model_flex_quant.tflite"
song_path = parent / "song.mp3"
model_path_str = str(model_path)
song_path_str = str(song_path)

sr = 44100

audio, _ = librosa.load(song_path_str, sr=sr, mono=False)
audio = audio.T.astype(np.float32)

interpreter = tf.lite.Interpreter(model_path=model_path_str, num_threads=1)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Expected shape:", input_details[0]['shape'])

if -1 in input_details[0].get('shape_signature', []):
    interpreter.resize_tensor_input(
        input_details[0]['index'],
        audio.shape,
        strict=False
    )
    interpreter.allocate_tensors()

inp_idx = input_details[0]['index']
inp_type = input_details[0]['dtype']
if inp_type == np.int8:
    scale, zp = input_details[0]['quantization']
    input_data = np.round(audio / scale + zp).astype(np.int8)
else:
    input_data = audio.astype(inp_type)

interpreter.set_tensor(inp_idx, input_data)
interpreter.invoke()

def dequantize(x, detail):
    s, z = detail['quantization']
    return (x.astype(np.float32) - z) * s

outputs = []
for detail in output_details:
    out = interpreter.get_tensor(detail['index'])
    if detail['dtype'] == np.int8:
        out = dequantize(out, detail)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out.squeeze(axis=0)
    outputs.append(out)

vocals, accompaniment = outputs

def to_int16(x):
    x = x / np.max(np.abs(x))
    return (x * np.iinfo(np.int16).max).astype(np.int16)

wav_output = str(parent / "tflites" / "outputs")

wav_write(wav_output + 'vocals.wav', sr, to_int16(vocals))
wav_write(wav_output + 'accompaniment.wav', sr, to_int16(accompaniment))
print("WAV files saved.")
