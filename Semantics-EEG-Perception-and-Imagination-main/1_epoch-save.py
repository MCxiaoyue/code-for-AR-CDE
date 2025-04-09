import sys
import mne
from mne import preprocessing
import pandas as pd
from mne.io import read_raw_eeglab
from mne.io import eeglab
import numpy as np
import os.path
from os import path
from autoreject import get_rejection_threshold

#  parameters
ppt_nums = ['3', '8']  # '15' str(i) for i in range(10, 20)
session = '3'

for ppt_num in ppt_nums:
    pathdata = 'sub' + ppt_num + '_sess' + session + '_50_ica_eeg.fif'
    xpath = 'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\sub-0' + ppt_num + '\\ses-0' + session + '\\eeg\\'
    pathdata = xpath + pathdata

    # Load up the raw .fif file
    # Reject Bad Epochs based on Automatic Thresholding
    raw = mne.io.read_raw_fif(pathdata)
    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    print("The events...", events)
    epochs = mne.Epochs(raw=raw, events=events, event_id=event_ids, preload=True, tmin=0, tmax=4, baseline=None,
                        event_repeated='merge')
    reject = get_rejection_threshold(epochs, ch_types='eeg', decim=2)
    epochs.drop_bad(reject=reject)
    print("The number of distinct events, not trials :",
          len(event_ids))  # this will give the number of distinct events, not the amount of trials
    print("The event names ", event_ids)

    # Grouping into Categories
    imag_flower_t = []
    imag_penguin_t = []
    imag_guitar_t = []
    imag_flower_p = []
    imag_guitar_p = []
    imag_penguin_p = []
    imag_flower_s = []
    imag_guitar_s = []
    imag_penguin_s = []
    perc_flower_t = []
    perc_guitar_t = []
    perc_penguin_t = []
    perc_flower_p = []
    perc_guitar_p = []
    perc_penguin_p = []
    perc_flower_s = []
    perc_guitar_s = []
    perc_penguin_s = []

    for id in event_ids:
        if 'Imagination' in id:
            if 'flower' in id:
                if '_t_' in id:
                    imag_flower_t.append(id)
                elif '_image_' in id:
                    imag_flower_p.append(id)
                elif '_a_' in id:
                    imag_flower_s.append(id)
                else:
                    print(id)
            elif 'guitar' in id:
                if '_t_' in id:
                    imag_guitar_t.append(id)
                elif '_image_' in id:
                    imag_guitar_p.append(id)
                elif '_a_' in id:
                    imag_guitar_s.append(id)
                else:
                    print(id)
            elif 'penguin' in id:
                if '_t_' in id:
                    imag_penguin_t.append(id)
                elif '_image_' in id:
                    imag_penguin_p.append(id)
                elif '_a_' in id:
                    imag_penguin_s.append(id)
                else:
                    print(id)

        if 'Perception' in id:
            if 'flower' in id:
                # print(id)
                if '_t_' in id:
                    perc_flower_t.append(id)
                elif '_image_' in id:
                    perc_flower_p.append(id)
                elif 'a_' in id or 'audio' in id:
                    perc_flower_s.append(id)
                else:
                    print(id)
            elif 'guitar' in id:
                if '_t_' in id:
                    perc_guitar_t.append(id)
                elif '_image_' in id:
                    perc_guitar_p.append(id)
                elif 'tiona_' in id or 'audio' in id:
                    perc_guitar_s.append(id)
                else:
                    print(id)
            elif 'penguin' in id:
                if '_t_' in id:
                    perc_penguin_t.append(id)
                elif '_image_' in id:
                    perc_penguin_p.append(id)
                elif 'tiona_' in id or 'audio' in id:
                    perc_penguin_s.append(id)
                else:
                    print(id)

    event_list = [imag_flower_t, imag_penguin_t, imag_guitar_t, imag_flower_p, imag_guitar_p,
                  imag_penguin_p, imag_flower_s, imag_guitar_s, imag_penguin_s,
                  perc_flower_t, perc_guitar_t, perc_penguin_t, perc_flower_p,
                  perc_guitar_p, perc_penguin_p, perc_flower_s, perc_guitar_s, perc_penguin_s]
    print("There should be 18 conditions: ", len(event_list))
    for li in event_list:
        print("Amount of different types in each condition ", len(li))

    # merge all the trials into their conditions, do this inplace to save on memory load

    mne.epochs.combine_event_ids(epochs, imag_flower_t, {'imag_flower_t': 300}, copy=False)
    mne.epochs.combine_event_ids(epochs, imag_penguin_t, {'imag_penguin_t': 301}, copy=False)
    mne.epochs.combine_event_ids(epochs, imag_guitar_t, {'imag_guitar_t': 302}, copy=False)

    mne.epochs.combine_event_ids(epochs, imag_flower_p, {'imag_flower_p': 303}, copy=False)
    mne.epochs.combine_event_ids(epochs, imag_penguin_p, {'imag_penguin_p': 304}, copy=False)
    mne.epochs.combine_event_ids(epochs, imag_guitar_p, {'imag_guitar_p': 305}, copy=False)

    mne.epochs.combine_event_ids(epochs, imag_flower_s, {'imag_flower_s': 306}, copy=False)
    mne.epochs.combine_event_ids(epochs, imag_penguin_s, {'imag_penguin_s': 307}, copy=False)
    mne.epochs.combine_event_ids(epochs, imag_guitar_s, {'imag_guitar_s': 308}, copy=False)

    mne.epochs.combine_event_ids(epochs, perc_flower_t, {'perc_flower_t': 309}, copy=False)
    mne.epochs.combine_event_ids(epochs, perc_penguin_t, {'perc_penguin_t': 310}, copy=False)
    mne.epochs.combine_event_ids(epochs, perc_guitar_t, {'perc_guitar_t': 311}, copy=False)

    mne.epochs.combine_event_ids(epochs, perc_flower_p, {'perc_flower_p': 312}, copy=False)
    mne.epochs.combine_event_ids(epochs, perc_penguin_p, {'perc_penguin_p': 313}, copy=False)
    mne.epochs.combine_event_ids(epochs, perc_guitar_p, {'perc_guitar_p': 314}, copy=False)

    mne.epochs.combine_event_ids(epochs, perc_flower_s, {'perc_flower_s': 315}, copy=False)
    mne.epochs.combine_event_ids(epochs, perc_penguin_s, {'perc_penguin_s': 316}, copy=False)
    mne.epochs.combine_event_ids(epochs, perc_guitar_s, {'perc_guitar_s': 317}, copy=False)

    # keep only new selection of epochs

    epochs_of_interest = ['imag_flower_t', 'imag_penguin_t', 'imag_guitar_t', 'imag_flower_p', 'imag_guitar_p',
                          'imag_penguin_p', 'imag_flower_s', 'imag_guitar_s', 'imag_penguin_s',
                          'perc_flower_t', 'perc_guitar_t', 'perc_penguin_t', 'perc_flower_p',
                          'perc_guitar_p', 'perc_penguin_p', 'perc_flower_s', 'perc_guitar_s', 'perc_penguin_s']
    epochs_sub = epochs[epochs_of_interest]

    # Editing the Durations

    imagine_orthographic_epochs = epochs_sub[['imag_flower_t', 'imag_guitar_t', 'imag_penguin_t']].crop(tmin=0, tmax=4)
    imagine_pictorial_epochs = epochs_sub[['imag_flower_p', 'imag_guitar_p', 'imag_penguin_p']].crop(tmin=0, tmax=4)
    imagine_audio_epochs = epochs_sub[['imag_flower_s', 'imag_guitar_s', 'imag_penguin_s']].crop(tmin=0, tmax=4)
    perception_orthographic_epochs = epochs_sub[['perc_flower_t', 'perc_guitar_t', 'perc_penguin_t']].crop(tmin=0,
                                                                                                           tmax=3)
    perception_pictorial_epochs = epochs_sub[['perc_flower_p', 'perc_guitar_p', 'perc_penguin_p']].crop(tmin=0, tmax=3)
    perception_audio_epochs = epochs_sub[['perc_flower_s', 'perc_guitar_s', 'perc_penguin_s']].crop(tmin=0, tmax=2)
    print(perception_audio_epochs, perception_orthographic_epochs, perception_pictorial_epochs, imagine_audio_epochs,
          imagine_orthographic_epochs, imagine_pictorial_epochs)

    # Save Epochs to file based on condition

    epochs_fname = 'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\'
    imagine_orthographic_epochs.save(epochs_fname + 'imagine_orthographic\\' + ppt_num + '_' + session + '_epo.fif',
                                     overwrite=True)
    imagine_pictorial_epochs.save(epochs_fname + 'imagine_pictorial\\' + ppt_num + '_' + session + '_epo.fif',
                                  overwrite=True)
    imagine_audio_epochs.save(epochs_fname + 'imagine_audio\\' + ppt_num + '_' + session + '_epo.fif', overwrite=True)
    perception_orthographic_epochs.save(
        epochs_fname + 'perception_orthographic\\' + ppt_num + '_' + session + '_epo.fif', overwrite=True)
    perception_pictorial_epochs.save(epochs_fname + 'perception_pictorial\\' + ppt_num + '_' + session + '_epo.fif',
                                     overwrite=True)
    perception_audio_epochs.save(epochs_fname + 'perception_audio\\' + ppt_num + '_' + session + '_epo.fif',
                                 overwrite=True)


