import streamlit as st
import os
from myfunctions import transcribing, downloader, sample_audio
import torch
torch.cuda.empty_cache()
speakers = {}

uploaded_file = st.file_uploader("Choose an audio file")
num_speakers = st.slider('How many people spoke in this audio? If unsure, enter the number of attendees in this meeting', min_value=1, max_value=10, value=2, step=1)
unique_speakers_dir = 'pyannote_transcripts'
number = 5

wav_files = [f for f in os.listdir(unique_speakers_dir) if f.endswith('.wav')]
if uploaded_file is not None:
    path = downloader(uploaded_file)
    
    if num_speakers is not None: 
        st.markdown(f'attempting to identify {num_speakers} speakers from audio...')
        print('num of speakers',num_speakers)
        identity_speaker = transcribing(path,num_speakers)
        wav_files = sample_audio(path, identity_speaker)
        with st.form(key='sweet_form'):
            for i, wav_file in enumerate(wav_files):
                col1, col2 = st.columns(2)
                col1.audio(os.path.join(unique_speakers_dir, wav_file), format='audio/wav')
                print(i)
                speaker_name = col2.text_input(f"Enter the name of the speaker", key = i)
                if speaker_name:
                    speakers[wav_file] = speaker_name
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.markdown(speakers)
                st.markdown('successfully identified speakers!')

