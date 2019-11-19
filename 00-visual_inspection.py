#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=================================
00. Visual inspection of raw data
=================================

Import raw data and plot.
If bad channels are marked in config, they will be added and shown on the plot. 
Currently, interactive plotting does not work, so if you want to add more bad channels,
you have to add them manually in the config, or your txt file. 

If running the scripts from a notebook or spyder
run %matplotlib qt in the command line to get the plots in extra windows

XXX: 
Make the plot interactive to allow to mark more bad channels.
Currently there is a proble with spyder, the script won't continue.
These will be saved in an annotations.csv file.
also implement rejection of bad segments (when epoching)

"""  # noqa: E501

import os.path as op

import mne
from mne.parallel import parallel_func
from warnings import warn

import config


def visual_inspection(subject):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    n_raws = 0
    for run in config.runs:

        # read bad channels for run from config
        if run:
            bads = config.bads[subject][run]
        else:
            bads = config.bads[subject]

        extension = run + '_raw'
        raw_fname_in = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))
        extension = run + '_annotations'
        annotations_fname_out = op.join(meg_subject_dir,
                                        config.base_fname.format(**locals()))
        pre, ext = op.splitext(annotations_fname_out)
        annotations_fname_out = pre + '.csv'

        print("Input: ", raw_fname_in)
        print("Output: None")
        # print("Output: ", annotations_fname_out)

        if not op.exists(raw_fname_in):
            warn('Run %s not found for subject %s ' %
                 (raw_fname_in, subject))
            continue

        raw = mne.io.read_raw_fif(raw_fname_in,
                                  allow_maxshield=config.allow_maxshield,
                                  preload=True, verbose='error')
        n_raws += 1
        # add bad channels
        raw.info['bads'] = bads
        print("added bads: ", raw.info['bads'])

        if config.set_channel_types is not None:
            raw.set_channel_types(config.set_channel_types)
        if config.rename_channels is not None:
            raw.rename_channels(config.rename_channels)
        raw.plot(n_channels=50, butterfly=True, group_by='position',
                 block=False)
#        raw.plot(n_channels=50, butterfly=True, group_by='position',
#                 block=True)

#        raw.annotations.save(annotations_fname_out)
#        print ("Annotations saved")
    if n_raws == 0:
        raise ValueError('No input raw data found.')


parallel, run_func, _ = parallel_func(visual_inspection, n_jobs=1)
parallel(run_func(subject) for subject in config.subjects_list)
