# in command prompt, run "python streamlit.py" to execute this file

import streamlit as st
import os
from myfunctions import transcribing, downloader, sample_audio, transcript_w_names, simple_transcript
import torch
import argparse
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--run_path', type=str, help='directory to store output')
args = parser.parse_args()
run_path = args.run_path


uploaded_file = st.file_uploader("Choose an audio file")
num_speakers = st.slider('How many people spoke in this audio? If unsure, enter the number of attendees in this meeting.', min_value=1, max_value=10, value=2, step=1)
unique_speakers_dir = 'unique_speakers'

if uploaded_file is not None:
    path = downloader(uploaded_file, run_path)
    
    if num_speakers is not None: 
        speakers = {}
        st.markdown(f'Attempting to identify {num_speakers} speakers in this audio...')
        print('num of speakers',num_speakers)
        identity_speaker, segments = transcribing(path, run_path, num_speakers)
        wav_files = sample_audio(path, run_path, identity_speaker)
        st.markdown(f'Please include the name of each speaker. If there are more than 1 speaker, only provide the name of the main speaker.')
        
        with st.form(key='sweet_form'):
            for i, wav_file in enumerate(wav_files):
                col1, col2 = st.columns(2)
                col1.audio(os.path.join(run_path,unique_speakers_dir, wav_file), format='audio/wav')
                speaker_name = col2.text_input(f"Enter the name of the speaker", key = i)
                if speaker_name:
                    speakers[wav_file[:-4]] = speaker_name
            submitted = st.form_submit_button("Submit")
            if submitted:
                #st.markdown(speakers)
                st.markdown('Speaker identification completed')
                transcript_path = simple_transcript(run_path, segments, speakers)
                print(speakers)
                with open(transcript_path) as file:
                    contents = file.read()
                    #st.code(contents, language = 'markdown')
                    st.markdown('Done!')
                    mytext = st.text_area('Kindly check if there are any mispelt arcronyms or nouns', value = str(contents), height = 400)
                    #st.write(f'Character count: {len(mytext)}')
                    st.write ('Thank you for using Convo2Calendar to transcribe your meetings. You may copy and paste the transcribed text into our tool to create formatted meeting minutes with your preferred word template.')

