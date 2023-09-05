import os
# import whisper
import speech_recognition
from pydub import AudioSegment, silence
from loguru import logger
import soundfile as sf
import numpy as np
from tqdm import tqdm


def main():
    # model = whisper.load_model("small")

    audio_path = "./inputs/sample1_trim2.wav"
    outdir = "./temp"
    script_dir = "./temp_script"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if not os.path.isdir(script_dir):
        os.makedirs(script_dir)

    recognizer = speech_recognition.Recognizer()

    audio = AudioSegment.from_wav(audio_path)
    ranges = silence.detect_silence(audio, min_silence_len=700, silence_thresh=audio.dBFS-16)
    idx = 0
    for x, y in tqdm(ranges):
        chunk = audio[idx:y]
        chunk_path = os.path.join(outdir, str(idx) + "_" + str(y) + ".wav")
        script_path = os.path.join(script_dir, str(idx) + "_" + str(y) + ".txt")
        chunk.export(chunk_path, format="wav")
        idx = y

        # result = model.transcribe(chunk_path, language="vi")
        # transcript = result["text"]
        # logger.success(f"transcript: {chunk_path}\n {transcript}")
        try:
            with speech_recognition.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language="vi-VN")

            logger.success(f"transcript: {chunk_path}\n {transcript}")
            with open(script_path, "w") as f:
                f.write(transcript)
        except:
            pass

    # try:
    #     with speech_recognition.AudioFile(audio_path) as source:
    #         audio_data = recognizer.record(source)
    #     transcript = recognizer.recognize_google(audio_data, language="vi-VN")

    #     logger.success(f"transcript: {transcript}")
    #     with open("./inputs/sample1_trim2.txt", "w") as f:
    #         f.write(transcript)
    # except:
    #     pass

if __name__ == "__main__":
    main()
