import os
from functools import partial
import torch
import hashlib
import math
from os.path import join as join_path
import random
import re
import glob
import time
from typing import Dict, List, Any

import pandas as pd
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset
import torchaudio
import torch.nn.functional as F

from .data_utils import SetDataset

SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0

class MSWCDataset:
    def __init__(self, args, cuda: bool=False):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration'] 
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)

        # main data dir
        self.data_dir = args['datadir']
        self.csv_file = args['csv_file']
        self.noise_dir = args['noise_dir']

        # add noise
        self.use_background = args['include_noise']
        self.background_volume = args['noise_snr']
        self.background_frequency= args['noise_frequency']
        if self.use_background:
            self.background_data = self.load_background_data()
            if self.background_data: print('Success load background data!')
            else: print('Failed load background data!')
        else:
            self.background_data = []
        
        self.use_wav = args['use_wav']
        self.cuda = cuda

        self.generate_data_dictionary()
        self.transforms = compose([
                partial(self.load_audio, 'file', 'label', 'data'),
                #partial(self.adjust_volume, 'data'),
                partial(self.shift_and_pad, 'data', 'file'),
                partial(self.mix_background, self.use_background,'data'),
                #partial(self.extract_features, 'data', 'feat'),
                partial(self.label_to_idx, 'label', 'label_idx')

        ])

    def generate_data_dictionary(self):
        self.data_set = defaultdict(list)
        with open(self.csv_file) as f:
            f.readline()
            for line in f:
                link, word, *rest = line.split(',', maxsplit=2)
                if self.use_wav: link = link.replace(".opus",".wav")
                self.data_set[word].append({'label': word, 'file': link})

        self.data_set = dict(self.data_set)
        self.word_to_index = {word: idx for idx, word in enumerate(self.data_set, 1)}


    def get_transform_dataset(self, file_dict, classes: List[str]):
        # classes is a list of classes
        file_dict = sum([file_dict[c] for c in classes], [])
        ls_ds = ListDataset(file_dict)
        ts_ds = TransformDataset(ls_ds, self.transforms)

        return ts_ds

    def get_dataloaders(self,
        npos: int, nneg: int, nshot: int,
        threshold: int,
        batch_size: int = 16,
        nworkers: int = 4,
    ):
        # make sure all random has the same seed
        random_state = random.getstate()

        random.setstate(random_state)
        wanted_words = random.sample(list(self.word_to_index), npos + nneg)
        pos_words = wanted_words[:npos]
        neg_words = wanted_words[npos:]

        ds_train: Dict[str, List[str]] = {}
        ds_test: Dict[str, List[str]] = {}

        for word in pos_words:
            random.setstate(random_state)
            files = random.sample(self.data_set[word], threshold)

            ds_train[word] = files[:nshot]
            ds_test[word] = files[nshot:]

        for word in neg_words:
            random.setstate(random_state)
            files = random.sample(self.data_set[word], threshold)
            ds_test[word] = files

        dl_list = [
            DataLoader(
                self.get_transform_dataset(ds_train, [word]),
                batch_size=nshot, num_workers=0,
            )
            for word in pos_words
        ]
        dl_train = DataLoader(
            SetDataset(dl_list),
            batch_size=npos, num_workers=nworkers,
            pin_memory=self.cuda,
        )

        dl_test = DataLoader(
            self.get_transform_dataset(ds_test, wanted_words),
            batch_size=batch_size, num_workers=nworkers,
            pin_memory=self.cuda,
            shuffle=True,
        )

        return dl_train, dl_test


    @property
    def num_classes(self):
        return len(self.word_to_index)
        
    def label_to_idx(self, k, key_out, d):
        label_index = self.word_to_index[d[k]]
        d[key_out] = torch.LongTensor([label_index]).squeeze()
        return d

    # --------------------AUDIO RELATED FUNCTIONS--------------------#

    def mix_background(self, use_background, k, d):
        if use_background and len(self.background_data) > 0: # add background noise as data augumentation
            foreground = d[k]
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            if len(background_samples) <= self.desired_samples:
                raise ValueError(
                    'Background sample is too short! Need more than %d'
                    ' samples but only %d were found' %
                    (self.desired_samples, len(background_samples)))
            background_offset = np.random.randint(
                0, len(background_samples) - self.desired_samples)
            background_clipped = background_samples[background_offset:(
                background_offset + self.desired_samples)]
            background_reshaped = background_clipped.reshape([1, self.desired_samples])
        
            if np.random.uniform(0, 1) < self.background_frequency:
                bg_snr = np.random.uniform(0, self.background_volume)
                s_pow = foreground.pow(2).sum()
                n_pow = background_reshaped.pow(2).sum()
                bg_vol = (s_pow/((10**bg_snr/10)*n_pow)).sqrt().item()
            else:
                bg_vol = 0

            background_mul = background_reshaped * bg_vol
            background_add = background_mul + foreground
            background_clamped = torch.clamp(background_add, -1.0, 1.0)
            d[k] = background_clamped

        return d
    
    def extract_features(self, k, key_out, d):
        # moved to the model

        if self.cuda:
            d_in = d[k].cuda()
        else:
            d_in = d[k]
        features = self.mfcc(d_in)[0] # just one channel
        features = torch.narrow(features, 0, 0, self.feature_bin_count)
        features = features.T # f x t -> t x f
        d[key_out] = torch.unsqueeze(features,0)

        return d

    def load_background_data(self):
        background_path = join_path(self.noise_dir, '*.wav')
        background_data = []
        for wav_path in glob.glob(background_path):
            bg_sound, bg_sr = torchaudio.load(wav_path)
            background_data.append(bg_sound.flatten())
        return background_data
    
    def build_mfcc_extractor(self):
        # moved to the model

        def next_power_of_2(x):  
            return 1 if x == 0 else 2**math.ceil(math.log2(x))
        
        frame_len = self.window_size_ms / 1000
        stride = self.window_stride_ms / 1000
        n_fft = next_power_of_2(frame_len*self.sample_rate)

        mfcc = torchaudio.transforms.MFCC(self.sample_rate,
                                        n_mfcc=self.n_mfcc ,
                                        log_mels = True,
                                        melkwargs={
                                            'win_length': int(frame_len*self.sample_rate),
                                            'hop_length' : int(stride*self.sample_rate),
                                            'n_fft' : int(frame_len*self.sample_rate),
                                            "n_mels": self.n_mfcc,
                                            "power": 2,
                                            "center": False                                         
                                        }
                                         )
        return mfcc

    def shift_and_pad(self, key, key_path, d):
        audio = d[key]

        audio =  audio * self.foreground_volume
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0
        
        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount
        
        
        # Ensure data length is equal to the number of desired samples
        audio_len = audio.size(1)
        if audio_len < self.desired_samples:
            pad = (0,self.desired_samples-audio_len)
            audio=F.pad(audio, pad, 'constant', 0) 

        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(padded_foreground, 1, time_shift_offset, self.desired_samples)
        d[key] = sliced_foreground

        return d

    
    def load_audio(self, key_path, key_label, out_field, d):
        filepath = join_path(self.data_dir, d[key_path])
        sound, sr = torchaudio.load(filepath, normalize=True)
        if sr != self.sample_rate:
            sound = torchaudio.functional.resample(sound, sr, self.sample_rate)

        # For silence samples, remove any sound
        if d[key_label] == SILENCE_LABEL:
             sound.zero_()

        d[out_field] = sound
        return d
