""" this is to extract text from smaller files and create output chunk
    files under data_text folder
"""
import json
import os

input_dir = 'data_smaller_files'
output_dir = 'data_text'

def extract_text_file(input_file, output_file):
    with open(input_file, 'r') as f:
        with open(output_file, 'w') as out:
            for line_number, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                    json.dump(data['text'], out)
                    out.write('\n')
                except json.JSONDecodeError as e:
                    # print(f"Error parsing JSON on line {line_number}: {e}")
                    continue


for file_name in os.listdir(input_dir):
    if file_name.endswith('.txt'):
        input_file = os.path.join(input_dir,file_name)
        output_file = os.path.join(output_dir,f'output_{file_name}')
        extract_text_file(input_file,output_file)
