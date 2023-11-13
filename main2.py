import streamlit as st
import os
from docxtpl import DocxTemplate
from myfunctions2 import *
import io

bio = io.BytesIO()

uploaded_file = st.file_uploader("Uploaded the word template that you wish to format your meeting notes. Note that it must be in Jinja2 templating format.")
transcript = st.text_area('Paste your meeting transcript here', height = 400)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    pipeline = model_setup()
    st.markdown('loading... 1/6 completed')
    transcript_str = load_transcript()
    st.markdown('loading... 2/6 completed')
    meeting_info_dict = meeting_info(transcript_str)
    st.markdown('loading... 3/6 completed')
    meeting_summary_text = meeting_summary(pipeline)
    st.markdown('loading... 4/6 completed')
    formatted_output = meeting_details(meeting_summary_text)
    st.markdown('loading... 5/6 completed')
    context = generate_table_content (meeting_info_dict,formatted_output)
    st.markdown('loading... 6/6 completed')
    st.markdown(context)
    if transcript is not None:
        with open('main2/user_upload/template.docx', 'wb') as f: 
            f.write(bytes_data)
        template = DocxTemplate("main2/user_upload/template.docx")
        template.render(context)
        template.save("main2/dynamic_table_out.docx")
    if os.path.exists("main2/dynamic_table_out.docx"):
        with open('main2/dynamic_table_out.docx', 'rb') as f:
            st.download_button('Download Formatted Meeting', f, file_name='main2/dynamic_table_out.docx')
    