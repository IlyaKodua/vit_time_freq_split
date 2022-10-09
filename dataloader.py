# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch


import numpy as np
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchaudio
import torch




class ESC_Dataset(Dataset):
    def __init__(self, dataset, esc_fold = 0, eval_mode = False):
        self.dataset = dataset
        self.esc_fold = esc_fold
        self.eval_mode = eval_mode
        self.sr = 32000
        random.seed(42)
        if self.eval_mode:
            self.dataset = self.dataset[self.esc_fold]
        else:
            temp = []
            for i in range(len(self.dataset)):
                if i != esc_fold:
                    temp += list(self.dataset[i]) 
            self.dataset = temp           
        self.total_size = len(self.dataset)
        self.queue = [*range(self.total_size)]
        if not eval_mode:
            self.generate_queue()

    def generate_queue(self):
        random.shuffle(self.queue)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "audio_name": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        p = self.queue[index]
        # data_dict = {
        #     "audio_name": self.dataset[p]["name"],
        #     "waveform": np.concatenate((self.dataset[p]["waveform"],self.dataset[p]["waveform"])),
        #     "real_len": len(self.dataset[p]["waveform"]) * 2,
        #     "target": self.dataset[p]["target"]
        # }

        wav = torch.from_numpy(self.dataset[p]["waveform"])[None,:].float()
        fbank = torchaudio.compliance.kaldi.fbank(wav, htk_compat=True, sample_frequency=self.sr, use_energy=False,
                                            window_type='hanning', num_mel_bins=128, dither=0.0, frame_length = 1024/self.sr*1000,
                                            frame_shift=498/self.sr*1000 )[None,:,:]
        target = torch.from_numpy(np.array([self.dataset[p]["target"]]))[:,0]
        return fbank, target

    def __len__(self):
        return self.total_size
