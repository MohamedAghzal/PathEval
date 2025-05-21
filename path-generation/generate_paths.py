import os
import subprocess
import sys
import random

directory = 'data'
n_samples = int(sys.argv[1])
files = os.listdir(directory)

l = int(sys.argv[2])
r = int(sys.argv[3])
algorithm = sys.argv[4]

for filename in files[l:r]:
    file_path = os.path.join(directory, filename)
    fp = open(f"progress{l}_{r}.txt", "a")
    
    if os.path.isfile(file_path):
            print(file_path)
            fp.write(f"Processing file: {file_path}\n")
            st = 0
            folder = f'paths-{algorithm.lower()}'
            if filename in os.listdir(folder):
                with open(f'{folder}/{filename}') as f:
                    data = f.read().split('@')
                    st = len(data) - 1

            for i in range(st, n_samples): 
                try:
                    result = subprocess.run(['./GeneratePaths', file_path, algorithm], capture_output=True, text=True)

                    if result.returncode == 0:
                        print(f"Successfully processed {filename} #{i}\n")
                        fp.write(f"Successfully processed {filename} #{i}\n")
                    else:
                        fp.write(f"Error processing {filename}")
                except Exception as e:
                    print(f"An error occurred while processing {filename} at try {i}: {e}")

