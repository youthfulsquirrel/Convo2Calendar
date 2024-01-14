import whisper
import datetime
from time import sleep
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.io import wavfile
import os
from pydub import AudioSegment
import librosa
import soundfile
import streamlit as st

def time_s(secs):
        return datetime.timedelta(seconds=round(secs))

def downloader(uploaded_file, run_path):
    audio_bytes = uploaded_file.read()
    download_path = os.path.join(os.getcwd(), run_path, 'user_upload', uploaded_file.name)
    # Write the contents of the uploaded file to a new file at the specified path
    with open(download_path, 'wb') as f:
        f.write(audio_bytes)
    sound = AudioSegment.from_mp3(download_path)
    sound.export(download_path[:-3]+'wav', format="wav")
    print(download_path[:-3])
    path = download_path[:-3] + 'wav'
    return (path)

def transcribing(path, run_path, num_speakers):
    embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda:1"))
    model = whisper.load_model("large", device="cuda:1")

    if path[-3:] != 'wav' and path[-3:] != 'mp3':
        print ('error !!!! ______________________')
    
    if path[-3:] == 'mp3':
        x,_ = librosa.load(path, sr=48000)
        soundfile.write(path, x, 48000)
        print('converting mp3 to wav')
        print(path)
    result = model.transcribe(path)
    segments = result["segments"]

    with contextlib.closing(wave.open(path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    audio = Audio()

    def segment_embedding(segment):
        start = segment["start"]
        # Whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(path, clip)
        return embedding_model(waveform[None])

    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment)

    embeddings = np.nan_to_num(embeddings)

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

    f = open(os.path.join(run_path,"transcript.txt"), "w")

    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time_s(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')
    f.close()
    identity_speaker = {}
    for (i, segment) in enumerate(segments):
        if i != 0 and segment["speaker"] not in identity_speaker:
            if i == 1 or segments[i - 1]["speaker"] != segment["speaker"]:
                start_time = int(segment["start"])
            if i+1 == len(segments) or segments[i + 1]["speaker"] != segment["speaker"]:
                end_time = int(segment["end"])
                identity_speaker[segment["speaker"]] = [start_time,end_time]
    return (identity_speaker, segments)

def sample_audio(path, run_path, identity_speaker):
    rate, data = wavfile.read(path) 
    unique_speakers_dir = 'unique_speakers'

    for speaker, speaker_time in identity_speaker.items():
        wavfile.write(os.path.join(run_path,unique_speakers_dir,speaker+'.wav'), rate, data[rate*speaker_time[0]+3:rate*(speaker_time[1]-3)])

    # Assuming that the 'unique_speakers' folder is in the same directory as your Streamlit script

    # List all .wav files in the 'unique_speakers' directory
    wav_files = [f for f in os.listdir(os.path.join(run_path,unique_speakers_dir)) if f.endswith('.wav')]
    return (wav_files)

def transcript_w_names(run_path, segments, speakers):
    print(run_path)
    with open(os.path.join(run_path,"final_transcript.txt"), "w") as f:
        print('printing segments')
        f.write('trial testing')
    # for (i, segment) in enumerate(segments):
    #     print(segments[i - 1]["speaker"])
    #     print(segment["speaker"])
    #     if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    #         print(segment["speaker"], '1')
    #         print(str(time_s(segment["start"])), '____2')
    #         f.write("\n" + segment["speaker"] + ' ' + str(time_s(segment["start"])) + '\n')
    #         f.write('testing testing...')
    #     f.write(segment["text"][1:] + ' ')
    # f.close()
    new_transcript = os.path.join(run_path,"final_transcript.txt")
    print('completed writing', new_transcript)
    return (os.path.join(run_path,"final_transcript.txt"))

def simple_transcript(run_path, segments, speakers):
    if not os.path.exists(os.path.join(run_path,"final_transcript.txt")):
        with open(os.path.join(run_path,"final_transcript.txt"), "w") as f:
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    f.write("\n" + speakers[segment["speaker"]] + ' ' + str(time_s(segment["start"])) + '\n')
                f.write(segment["text"][1:] + ' ')
    return (os.path.join(run_path,"final_transcript.txt"))
