import argparse
import shutil
import torch
import tqdm
import os
import numpy as np
import math
import cv2
import librosa
import pickle
import matplotlib.pyplot as plt
from util.util import save_pickle

AUDIO_DATA_PATH_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/Parallel_speech_data'
SUBDIRECTORIES_DEFAULT = ['clean','noisy']
CACHE_DEFAULT = '/content/Pix2Pix-VC/data_cache'
SAMPLING_RATE = 22050

## mel function inspired from https://github.com/GANtastic3/MaskCycleGAN-VC

def mel(wavspath):
    info = {
        'records_count' : 0, #Keep track of number of clips
        'duration' : 0 # Keep track of duration of training set for the speaker
    }
    
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    mel_list = list()
    wav_files = os.listdir(wavspath)
    filenames = []
    for file in wav_files:
        wavpath = os.path.join(wavspath,file)
        if wavpath[-4:] != '.wav':
            continue
        wav_orig, _ = librosa.load(wavpath, sr=SAMPLING_RATE, mono=True)
        spec = vocoder(torch.tensor([wav_orig]))
        #print(f'Spectrogram shape: {spec.shape}')
        if spec.shape[-1] >= 64:    # training sample consists of 64 frames
            mel_list.append(spec.cpu().detach().numpy()[0])
            info['duration']+=librosa.get_duration(filename=wavpath)
            info['records_count']+=1   
            filenames.append(file)       
    return mel_list, filenames, info





def preprocess_dataset_spectrogram(data_path, class_id, args):
    """Preprocesses dataset of .wav files by converting to Mel-spectrograms.
    Args:
        data_path (str): Directory containing .wav files of the speaker.
        speaker_id (str): ID of the speaker.
        cache_folder (str, optional): Directory to hold preprocessed data. Defaults to './cache/'.
    """

    print(f"Preprocessing data for class: {class_id}.")

    cache_folder = args.data_cache

    mel_list, filenames, info = mel(data_path)

    if not os.path.exists(os.path.join(cache_folder, class_id)):
        os.makedirs(os.path.join(cache_folder, class_id, 'meta'))
        os.makedirs(os.path.join(cache_folder, class_id, 'train'))
        os.makedirs(os.path.join(cache_folder, class_id, 'val'))
        os.makedirs(os.path.join(cache_folder, class_id, 'test'))
    
    indices = np.arange(0,len(mel_list))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_split = math.floor(args.train_percent/100*len(mel_list))
    val_split = math.floor(args.val_percent/100*len(mel_list))
    
    padding ={}

    for phase,(start, end) in zip(['train','val','test'],[(0,train_split),(train_split,train_split+val_split),(train_split+val_split,len(mel_list))]):
        if phase=='train':
            ## Get mean and norm
            train_samples=mel_list[start:end]
            mel_concatenated = np.concatenate(train_samples, axis=1)
            mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
            mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9
            np.savez(os.path.join(cache_folder, class_id, 'meta', f"{class_id}_stat"),mean=mel_mean,std=mel_std)

        for i in range(start,end):
            filename=filenames[indices[i]]
            img = (mel_list[indices[i]]-mel_mean)/mel_std
            filename=filename[:32]+'.pickle' ## THIS STEP IS SPECIFIC TO THE CURRENT DATASET TO ENSURE A AND B HAVE SAME FILENAMES
            
            ##Padding the image
            freq_len,time_len = img.shape
            top_pad = args.size_multiple - freq_len % args.size_multiple if freq_len % args.size_multiple!=0 else 0
            right_pad = args.size_multiple - time_len % args.size_multiple if time_len % args.size_multiple!=0 else 0
            x_size = time_len+right_pad
            y_size = freq_len+top_pad
            img_padded = np.zeros((y_size,x_size))
            img_padded[-freq_len:,0:time_len] = img

            ## Saving Padding info
            padding[filename[:-7]] = (top_pad,right_pad)

            ##Saving Image
            save_pickle(variable=img_padded,fileName=os.path.join(cache_folder,class_id,phase,filename))
        

    
    save_pickle(variable=padding,fileName=os.path.join(cache_folder, class_id, 'meta', f"{class_id}_padding.pickle"))

    print('#'*25)
    print(f"Preprocessed and saved data for class: {class_id}.")
    print(f"Total duration of dataset for {class_id} is {info['duration']} seconds")
    print(f"Total clips in dataset for {class_id} is {info['records_count']}")
    print('#'*25)

def get_filenames(fileNameA):
    """
    Custom function for this specific dataset.
    It returns the names of corresponding files in the 2 classes along with the common name by which it should be saved.

    Args:
    fileNameA(str) : Filename in the first class
    """

    return fileNameA, fileNameA[:32]+'-A.wav', fileNameA[:32]+'.wav'



def transfer_aligned_audio_raw(root_dir,class_ids,data_cache,train_percent,val_percent):
    """
    Transfer audio files to a convinient location for processing with train,test,validation split.
    Arguments:
    root_dir(str) - Root directory where files of specified classes are present in subdirectories.
    class_id(str) - Current class ID of data objects
    data_cache(str) - Root directory to store data
    train_percent(int) - Percent of data clips in train split
    val_percent(int) - Percent of data clips in validation split 
    """

    for class_id in class_ids:
        os.makedirs(os.path.join(data_cache,class_id,'train'))
        os.makedirs(os.path.join(data_cache,class_id,'val'))
        os.makedirs(os.path.join(data_cache,class_id,'test'))

    files_list = [x for x in os.listdir(os.path.join(root_dir,class_ids[0])) if x[-4:]=='.wav']
    num_files = len(files_list)

    indices = np.arange(0,num_files)
    np.random.seed(7)
    np.random.shuffle(indices)

    train_split = math.floor(train_percent/100*num_files)
    val_split = math.floor(val_percent/100*num_files)

    for phase,(start, end) in zip(['train','val','test'],[(0,train_split),(train_split,train_split+val_split),(train_split+val_split,num_files)]):
        duration = 0
        clips = 0
        for i in range(start,end):
            fileA, fileB, file=get_filenames(files_list[indices[i]])
            shutil.copyfile(os.path.join(root_dir,class_ids[0],fileA),os.path.join(data_cache,class_ids[0],phase,file))
            shutil.copyfile(os.path.join(root_dir,class_ids[1],fileB),os.path.join(data_cache,class_ids[1],phase,file))
            duration+=librosa.get_duration(filename=os.path.join(data_cache,class_ids[0],phase,file))
            clips+=1
        print(f'{duration} seconds ({clips} clips) of Audio saved to {phase}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Spectrograms')
    parser.add_argument('--audio_data_path', dest = 'audio_path', type=str, default=AUDIO_DATA_PATH_DEFAULT, help="Path to audio root folder")
    parser.add_argument('--source_sub_directories', dest = 'sub_directories',type=str, default=SUBDIRECTORIES_DEFAULT, help="Sub directories for data")
    parser.add_argument('--data_cache', dest='data_cache', type=str, default=CACHE_DEFAULT, help="Directory to Store data and meta data.")
    parser.add_argument('--train_percent', dest='train_percent', type=int, default=70, help="Percentage for train split")
    parser.add_argument('--val_percent', dest='val_percent', type=int, default=15, help="Percentage for val split")
    parser.add_argument('--size_multiple', dest='size_multiple', type=int, default=4, help="Required Factor of Dimensions")
    parser.add_argument('--transfer_mode', dest='transfer_mode', type=str, choices=['audio','spectrogram'], default='audio', help='Transfer files as raw audio or converted spectrogram.')
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))
    if args.transfer_mode == 'spectrogram':
        for class_id in args.sub_directories:        
            preprocess_dataset_spectrogram(os.path.join(args.audio_path,class_id),class_id,args)
    else:
        transfer_aligned_audio_raw(args.audio_path,args.sub_directories,args.data_cache,args.train_percent,args.val_percent)




