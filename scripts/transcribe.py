import streamlit as st
import os
import whisperx
import gc 
import datetime
from scipy.io import wavfile
from pydub import AudioSegment
from myfunctions import transcribing, downloader, sample_audio, simple_transcript
import torch
import argparse
torch.cuda.empty_cache()
# parser = argparse.ArgumentParser()
# parser.add_argument('--run_path', type=str, help='directory to store output')
# args = parser.parse_args()
# run_path = args.run_path

run_path = "../runs/Part1/run4"

with open(r'../keys/hugging_face.txt', 'r') as fp:
    # read all lines using readline()
    lines = fp.readlines()
    for line in lines:
        HF_TOKEN = line

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


def whisperx_transcribe(audio_file, max_speakers):
    language = 'en'
    device = "cuda" 
    #audio_file = "../sample_audio/sg_parliament_20min.mp3"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    #Transcribe with original whisper (batched)
    model = whisperx.load_model("small", device, compute_type=compute_type, language = language)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language = language)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio, max_speakers = max_speakers)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs
    return result["segments"]

def sample_audio(path, run_path, segments):
    rate, data = wavfile.read(path) 
    unique_speakers_dir = 'unique_speakers'

    identity_speaker = {}
    for (i, segment) in enumerate(segments):
        if i != 0 and segment["speaker"] not in identity_speaker:
            if i == 1 or segments[i - 1]["speaker"] != segment["speaker"]:
                start_time = int(segment["start"])
            if i+1 == len(segments) or segments[i + 1]["speaker"] != segment["speaker"]:
                end_time = int(segment["end"])
                identity_speaker[segment["speaker"]] = [start_time,end_time]

    for speaker, speaker_time in identity_speaker.items():
        wavfile.write(os.path.join(run_path,unique_speakers_dir,speaker+'.wav'), rate, data[rate*speaker_time[0]+3:rate*(speaker_time[1]-3)])

    # Assuming that the 'unique_speakers' folder is in the same directory as your Streamlit script

    # List all .wav files in the 'unique_speakers' directory
    wav_files = [f for f in os.listdir(os.path.join(run_path,unique_speakers_dir)) if f.endswith('.wav')]
    speaker_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(run_path, unique_speakers_dir)) if f.endswith('.wav')]
    speakers = {speaker_id: speaker_id for speaker_id in speaker_ids}
    return (wav_files, speakers)
    
def simple_transcript(run_path, segments, speakers):
    def time_s(secs):
        return datetime.timedelta(seconds=round(secs))
    if not os.path.exists(os.path.join(run_path,"final_transcript.txt")):
        with open(os.path.join(run_path,"final_transcript.txt"), "w") as f:
            for (i, segment) in enumerate(segments):
                print('-------------_____________________________')
                print(segment)
                print('___________________________-------')
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    print(speakers[segment["speaker"]])
                    print('+++')
                    print(segment["start"])
                    f.write("\n" + speakers[segment["speaker"]] + ' ' + str(time_s(segment["start"])) + '\n')
                f.write(segment["text"][1:] + ' ')
    return (os.path.join(run_path,"final_transcript.txt"))


def main():
    uploaded_file = st.file_uploader("Choose an audio file")
    num_speakers = st.slider('How many people spoke in this audio? If unsure, enter the number of attendees in this meeting.', min_value=1, max_value=10, value=2, step=1)
    unique_speakers_dir = 'unique_speakers'

    if uploaded_file is not None:
        path = downloader(uploaded_file, run_path)
        
        if num_speakers is not None: 
            st.markdown(f'Attempting to identify {num_speakers} speakers in this audio...')
            print('num of speakers',num_speakers)
            segments = whisperx_transcribe(path, num_speakers)
            wav_files, speakers = sample_audio(path, run_path, segments)
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

main()