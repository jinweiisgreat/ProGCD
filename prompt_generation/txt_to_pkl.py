import os
import json
from natsort import natsorted
from tqdm import tqdm
import pickle

def txt_to_pkl(txt_path,pkl_save_path):
    arr =[]
    with open(txt_path,'r') as txt_file:
        for data in txt_file:
            arr.append(data.replace("\n",""))
    txt_file.close()
    with open(pkl_save_path, "wb") as f:
        pickle.dump(arr, f)

# change .txt file to .pkl file
def main():
    dataset_name = 'cifar100'
    txt_to_pkl(f'./txt/{dataset_name}_a_photo_of_label.txt',f"./pkl/{dataset_name}_a_photo_of_label.pkl")
    

if __name__ == "__main__":
    main()