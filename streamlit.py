import os
import subprocess

def create_run_dir():
   i = 1
   while os.path.exists(f"run{i}"):
       i += 1
   dir_name = f"run{i}"
   return dir_name

dir_name = create_run_dir()
os.makedirs(dir_name, exist_ok=True)
os.makedirs(os.path.join(dir_name, 'user_upload'), exist_ok=True)
os.makedirs(os.path.join(dir_name, 'unique_speakers'), exist_ok=True)

# Get the path of the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to my_streamlit.py
streamlit_script_path = os.path.join(current_dir, 'main.py')

# Run the Streamlit script
subprocess.run(['streamlit', 'run', streamlit_script_path,'--', f'--run_path={dir_name}'])