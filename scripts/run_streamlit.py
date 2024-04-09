import os
import subprocess

def create_run_dir():
   i = 1

   while os.path.exists(os.path.join('..', 'runs', 'Part1', f"run{i}")):
       i += 1
   dir_name = os.path.join('..', 'runs', 'Part1', f"run{i}")
   return dir_name

def create_subfolders():
    os.makedirs(dir_name, exist_ok=True)
    print(f'created folder {os.path.abspath(dir_name)}')
    os.makedirs(os.path.join(dir_name, 'user_upload'), exist_ok=True) #user input
    os.makedirs(os.path.join(dir_name, 'unique_speakers'), exist_ok=True) #user input
    os.makedirs(os.path.join(dir_name, 'metadata'), exist_ok=True) #system generated output

    # Get the path of the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to my_streamlit.py
    streamlit_script_path = os.path.join(current_dir, 'main.py')

    # Run the Streamlit script
    subprocess.run(['streamlit', 'run', streamlit_script_path, '--server.fileWatcherType', 'none','--', f'--run_path={dir_name}'])

if __name__ == '__main__':
    dir_name = create_run_dir()
    create_subfolders()

    