import streamlit as st
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
from streamlit_autorefresh import st_autorefresh
#st_autorefresh(interval=10000, limit=100, key="fizzbuzzcounter")
torch.cuda.empty_cache()
from myfunctions import transcribing

print('initializing')
num_speakers = None

uploaded_file = st.file_uploader("Choose an audio file")
while uploaded_file is None:
    print('sleep1')
    sleep(2)
print('awake1')
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    download_path = os.path.join(os.getcwd(), uploaded_file.name)
    # Write the contents of the uploaded file to a new file at the specified path
    with open(download_path, 'wb') as f:
        f.write(audio_bytes)

    num_speakers = st.slider('How many people spoke in this audio? If unsure, enter the number of attendees in this meeting', min_value=1, max_value=10, value=2, step=1)
while num_speakers is None:
    sleep(2)
if num_speakers != None and st.button('start'):
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown("Processing the audio file...")
    print ('__________________________________________________________')
    print(download_path)
    sound = AudioSegment.from_mp3(download_path)
    sound.export(download_path[:-3]+'wav', format="wav")
    print(download_path[:-3])
    path = download_path[:-3] + 'wav'
    identity_speaker = transcribing(path,num_speakers)

    spinner_placeholder.text("Processing complete!")
    # read the file and get the sample rate and data
    rate, data = wavfile.read(path) 

    unique_speakers_dir = 'pyannote_transcripts'

    for speaker, speaker_time in identity_speaker.items():
        wavfile.write(os.path.join(unique_speakers_dir,speaker+'.wav'), rate, data[rate*speaker_time[0]+2:rate*(speaker_time[1]-2)])

    # Assuming that the 'unique_speakers' folder is in the same directory as your Streamlit script

    # List all .wav files in the 'unique_speakers' directory
    wav_files = [f for f in os.listdir(unique_speakers_dir) if f.endswith('.wav')]
    speakers = {}
    # Display each .wav file using st.audio
    st.markdown('Enter the name of the speaker in each audio file. If there is more than 1 speaker, only provide the name of the main speaker.')
    captured = False
    with st.form(key='my_form', clear_on_submit=False):
        for wav_file in wav_files:
            # Create two columns side by side
            col1, col2 = st.columns(2)

            # Display the audio player in the first column
            col1.audio(os.path.join(unique_speakers_dir, wav_file), format='audio/wav')

            # Display the text input widget in the second column
            speaker_name = col2.text_input(f"Enter the name of the speaker for {wav_file}")

            # Store the speaker name in the dictionary
            if speaker_name:
                speakers[wav_file] = speaker_name
        submitted = st.form_submit_button("Submit")
        while not captured:
            sleep(5)
            print(submitted)
            if submitted:
                captured = True
                with st.form(key='my_form2', clear_on_submit=False):
                    for wav_file in wav_files:
                    # Create two columns side by side
                        col3, col4 = st.columns(2)
                        # Display the audio player in the first column
                        col3.audio(os.path.join(unique_speakers_dir, wav_file), format='audio/wav')
                        # Display the text input widget in the second column
                        col4.text_input(f"Enter the name of the speaker for {wav_file}", speakers[wav_file])
                    resubmitted = st.form_submit_button("Re-submit")
                st.markdown('transcribing...')