import os
import random
import shutil

# Paths
src_dir = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=16"
dst1_dir = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=16v1"
dst2_dir = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=16v2"

# Create destination directories if they don't exist
os.makedirs(dst1_dir, exist_ok=True)
os.makedirs(dst2_dir, exist_ok=True)

# Get all file names (assuming flat structure, not recursive)
all_files = os.listdir(src_dir)
total_files = len(all_files)

# Shuffle the list
random.shuffle(all_files)

# Split (roughly) in half
midpoint = total_files // 2
files1 = all_files[:midpoint]
files2 = all_files[midpoint:]

# Move files
def move_files(file_list, dest_dir):
    for i, filename in enumerate(file_list):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(src_path, dest_path)
        if i % 100000 == 0:
            print(f"Moved {i} files to {dest_dir}")

print("Moving to N=16v1...")
move_files(files1, dst1_dir)

print("Moving to N=16v2...")
move_files(files2, dst2_dir)

print("Done!")
