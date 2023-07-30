import os
import openai
from pydub import AudioSegment
import time
import pickle
import hashlib

def md5_file(filename):
    """Calculates the MD5 checksum of a file."""
    with open(filename, "rb") as file:
        md5 = hashlib.md5()
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def transcribe_podcast(podcast_path, target_chunk_length = 25 * 1000, overlap = 0, max_retries = 3, retry_delay = 5, save_interval = 10):
    print("Transcribing podcast...")
    audio = AudioSegment.from_file(podcast_path)
    audio_length = len(audio)
    print("Total size of audio:", audio_length)

    md5 = md5_file(podcast_path)

    num_chunks = int(audio_length / target_chunk_length) + 1
    actual_chunk_length = audio_length / num_chunks

    directory, filename = os.path.split(podcast_path)
    state_file = os.path.join(directory, md5 + "_transcription_state.pkl")

    if os.path.isfile(state_file):
        with open(state_file, "rb") as f:
            state = pickle.load(f)
            start = state["start"]
            transcription = state["transcription"]
            print("Resuming from previous state...")
    else:
        start = 0
        transcription = ""

    counter = 0

    while start < audio_length:
        end = int(start + actual_chunk_length)
        chunk = audio[start:end]
        chunk.export("/tmp/chunk.wav", format="wav")

        chunk_transcript = None
        for i in range(max_retries):
            try:
                print("Transcribing chunk (Attempt", i + 1, "out of", max_retries, ")...")
                with open("/tmp/chunk.wav", "rb") as audio_file:
                    response = openai.Audio.transcribe(model="whisper-1", file=audio_file,
                                                       api_key=os.getenv('OPENAI_API_KEY'), output_format="text")
                chunk_transcript = response.get('text') 
                if chunk_transcript is not None:
                    break
                else:
                    print("Chunk transcription not available.")
            except openai.error.APIError as e:
                print("API Error occurred:", str(e))
                if i < max_retries - 1:
                    print("Retrying in", retry_delay, "seconds...")
                    time.sleep(retry_delay)

        if chunk_transcript is not None:
            transcription += chunk_transcript + " "
        else:
            print("Transcription failed for chunk. Retrying the chunk...")
            continue

        start = end - overlap
        completed_percentage = (end * 100.0) / audio_length
        print("Completed {:.2f}%".format(completed_percentage))

        counter += 1

        if counter % save_interval == 0:
            state = {"start": start, "transcription": transcription}
            with open(state_file, "wb") as f:
                pickle.dump(state, f)
                print("State saved at position", start)

    print("Podcast transcription complete.")
    return transcription

