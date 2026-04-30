import os
from functools import partial
import torch
import hashlib
import math
import os.path
import random
import re
import glob
import time

import pandas as pd
import soundfile as sf
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset
import torchaudio
import torch.nn.functional as F


from .data_utils import SetDataset

class EpisodicFixedBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fixed_silence_unknown = False):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.sampling = [torch.randperm(self.n_classes)[:self.n_way] for i in range(self.n_episodes)]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]
            
def partition(L, ratio):
    ratio = np.cumsum(ratio) * len(L) / sum(ratio)
    ratio = np.round(ratio).astype(int)
    
    partitions = []
    prev = None
    for r in ratio:
        partitions.append(L[prev:r])
        prev = r

    return partitions

SEED = 42
RATIO = [2, 1, 4]
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0

class MSWCDataset:
    def __init__(self, data_dir, MSWCtype, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration'] 
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)

        assert MSWCtype in {'pos', 'neg'}
        self.task = MSWCtype

        # main data dir
        self.data_dir = data_dir
        self.csv_dir = args['csvdir']
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
        self.unknown = args['include_unknown']
        self.silence = self.task == 'neg'

        self.unknown_ratio = 0.1
        self.silence_ratio = 0.1
        self.generate_data_dictionary()

    def generate_data_dictionary(self):
        # Prepare data sets
        self.data_set = {
            'training': defaultdict(list),
            'validation': defaultdict(list),
            'testing': defaultdict(list)
        }
        unknown_set = {'training': [], 'validation': [], 'testing': []}
        for split in ["training","validation","testing"]:
            # parse the right file
            if split == 'training':
                split_name = 'train'
            elif split == 'validation':
                split_name = 'dev'
            elif split == 'testing':
                split_name = 'test'
            df = pd.read_csv(self.csv_dir+split_name+".csv")

            if split == "training":
                all_words = {}
                for word in df['WORD']: all_words[word] = 1
                all_words = sorted(all_words)
                
                random.seed(SEED)
                random.shuffle(all_words)
                pos, unknown, neg = partition(all_words, RATIO)
                if self.task == 'pos':
                    wanted_words = pos
                    unknown_words = unknown
                else:
                    wanted_words = neg
                    unknown_words = []

                unknown_words = set(unknown_words)
                # build word to index
                skip = 0
                if self.silence: skip +=1
                if self.unknown: skip +=1
                else:
                    global SILENCE_INDEX
                    SILENCE_INDEX = SILENCE_INDEX -1

                self.word_to_index = {}
                if self.silence:
                    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
                if self.unknown:
                    self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX
                for idx, word in enumerate(wanted_words):
                    self.word_to_index[word] = idx + skip

            # build the dict dataset split
            for word, link in zip(df['WORD'], df['LINK']):
                if self.use_wav: link = link.replace(".opus",".wav")
                if word in self.word_to_index:
                    self.data_set[split][word].append({'label': word, 'file': link})
                elif word in unknown_words:
                    unknown_set[split].append({'label': UNKNOWN_WORD_LABEL, 'file': link})

             # Add silence and unknown words to each set
            set_size = len(sum(self.data_set[split].values(), []))
            if self.unknown and unknown_set[split]:
                unknown_size = min(
                    int(math.ceil(set_size * self.unknown_ratio)),
                    len(unknown_set[split])
                )
                random.seed(SEED)
                self.data_set[split][UNKNOWN_WORD_LABEL] = random.sample(
                    unknown_set[split], unknown_size)

            if self.silence:
                silence_path = df.iloc[0]['LINK']
                silence_size = int(math.ceil(set_size * self.silence_ratio))
                self.data_set[split][SILENCE_LABEL] = [
                    {'label': SILENCE_LABEL, 'file': silence_path}
                    for _ in range(silence_size)
                ]

            del df

        for split, d in self.data_set.items():
            self.data_set[split] = dict(d)
        
    def get_transform_dataset(self, file_dict, classes, filters=None):
        # classes is a list of classes
        transforms = compose([
                partial(self.load_audio, 'file', 'label', 'data'),
                #partial(self.adjust_volume, 'data'),
                partial(self.shift_and_pad, 'data', 'file'),
                partial(self.mix_background, self.use_background,'data'),
                #partial(self.extract_features, 'data', 'feat'),
                partial(self.label_to_idx, 'label', 'label_idx')

        ])
        file_dict = sum([file_dict[c] for c in classes], [])
        ls_ds = ListDataset(file_dict)
        ts_ds = TransformDataset(ls_ds, transforms)
        
        return ts_ds

    def get_episodic_fixed_sampler(self, num_classes,  n_way, n_episodes, fixed_silence_unknown = False):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes, fixed_silence_unknown = fixed_silence_unknown)    
    
    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic'):
        if set_index not in ['training', 'validation', 'testing']:
            raise ValueError("Set index = {} in episodic dataset is not correct.".format(set_index))

        dataset = self.data_set[set_index]
        if sampler == 'episodic':
            # load all possible classes
            sampler = self.get_episodic_fixed_sampler(len(dataset), len(dataset), n_episodes)

        dl_list=[]        
        for k, keyword in enumerate(dataset):
            ts_ds = self.get_transform_dataset(dataset, [keyword])

            if n_samples <= 0:
                n_samples = len(ts_ds)

            dl = torch.utils.data.DataLoader(ts_ds, batch_size=n_samples,
                    shuffle=True, num_workers=0)
            dl_list.append(dl)

        ds = SetDataset(dl_list)
        data_loader_params = dict(batch_sampler = sampler, num_workers=8,
                pin_memory=not self.cuda)
        dl = torch.utils.data.DataLoader(ds, **data_loader_params)

        return dl
    
    
    def get_iid_dataloader(self, split, batch_size, class_list=[]):
        if not class_list: class_list = list(self.data_set[split].keys())
        ts_ds = self.get_transform_dataset(self.data_set[split], class_list)
        dl = torch.utils.data.DataLoader(
            ts_ds, batch_size=batch_size,
            pin_memory=not self.cuda, shuffle=True, num_workers=8
        )

        return dl
    
    def num_classes(self):
        return len(self.word_to_index)
        
    def label_to_idx(self, k, key_out, d):
        label_index = self.word_to_index[d[k]]
        d[key_out] = torch.LongTensor([label_index]).squeeze()

        return d

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
        background_path = os.path.join(self.noise_dir, '*.wav')
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
        filepath = os.path.join(self.data_dir, d[key_path]) # d[key_path] -> LINK entry in split file
        sound, sr = torchaudio.load(filepath, normalize=True)
        if sr != self.sample_rate:
            sound = torchaudio.functional.resample(sound, sr, self.sample_rate)

        # For silence samples, remove any sound
        if d[key_label] == SILENCE_LABEL:
             sound.zero_()

        d[out_field] = sound
        return d
