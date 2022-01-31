import argparse
import os
import cv2
import numpy as np
import torch
import librosa
import torchaudio
import pickle

RESULTS_PATH_DEFAULT = '/content/Pix2Pix-VC/results/noise_pix2pix/test_latest/images'
SAVE_PATH_DEFAULT = '/content/Pix2Pix-VC/data_cache/results' 
PADDING_PATH_DEFAULT = '/content/Pix2Pix-VC/data_cache/noisy/meta/noisy_padding.pickle'
SAMPLING_RATE = 22050


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def get_audio(data_root,padding_path,save_path):
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    padding = load_pickle_file(padding_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(data_root):
        if file[-4:]!= '.png':
            continue
        img_padded = np.array(cv2.imread(os.path.join(data_root,file),0))
        key = file[:33]+'.jpg'
        top_pad,right_pad = padding[key]
        img = img_padded[top_pad:,0:img_padded.shape[1]-right_pad]
        img =torch.tensor(img,dtype=torch.float32)
        rev = vocoder.inverse(img.unsqueeze(0)).cpu().detach()
        torchaudio.save(os.path.join(save_path,file[:-4]+'.wav'), rev, sample_rate=SAMPLING_RATE, bits_per_sample=16)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Audiofiles from spectrograms')
    parser.add_argument('--results_path', dest = 'results_path', type=str, default=RESULTS_PATH_DEFAULT, help="Path to results folder")
    parser.add_argument('--save_path', dest='save_path', type=str, default=SAVE_PATH_DEFAULT, help="Directory to save Audio files.")
    parser.add_argument('--padding_path', dest='padding_path', type=str, default=PADDING_PATH_DEFAULT, help="Path to padding data")
    args = parser.parse_args()
    get_audio(args.results_path,args.padding_path,args.save_path)