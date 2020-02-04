"""
===================================
01. Maxwell filter using MNE-Python
===================================

The data are imported for the first time.
If you specified annotations (using script 00_visual_inspection.py),
the bad channels will be read from there.

If  use_maxwell_filter=True in config, maxfilter will be applied,
 using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.
Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.
The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.
The function loads machine-specific calibration files from the paths set for
config.mf_ctc_fname  and config.mf_cal_fname.

A file with the extension _sss_raw.fif will be saved per subject and run.
If  use_maxwell_filter=False in config, maxfilter will be NOT applied.
A file with the extension _nosss_raw.fif will be saved per subject and run.
Important: If your data were recorded with internal active compensation
(MaxShield), you have to maxfilter them, otherwise they will be distorted.

If config.plot = True plots raw data.
"""  # noqa: E501

import os.path as op

import mne
from mne.parallel import parallel_func
from warnings import warn

import config


def run_maxwell_filter(subject):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    n_raws = 0
    for run in config.runs:

        extension = run + '_raw'
        raw_fname_in = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))

        if config.use_maxwell_filter:
            extension = run + '_sss_raw'
        else:
            extension = run + '_nosss_raw'

        raw_fname_out = op.join(meg_subject_dir,
                                config.base_fname.format(**locals()))

        print("Input: ", raw_fname_in)
        print("Output: ", raw_fname_out)

        if not op.exists(raw_fname_in):
            warn('Run %s not found for subject %s ' %
                 (raw_fname_in, subject))
            continue

        # read raw data
        raw = mne.io.read_raw_fif(raw_fname_in,
                                  allow_maxshield=config.allow_maxshield,
                                  preload=True, verbose='error')

        # add bad channels, using the annotations if found
        annotations_fname_out = op.join(meg_subject_dir,
                                        config.base_fname.format(**locals()))
        pre, ext = op.splitext(annotations_fname_out)
        annotations_fname_in = pre + '.csv'

        if op.exists(annotations_fname_in):
            annotations = mne.read.annotations(annotations_fname_in)
            raw.set_annotations(annotations)
            print("Reading bads from annotations")
            print("bads: ", raw.info['bads'])
        else:
            # read bad channels for run from config
            if run:
                bads = config.bads[subject][run]
            else:
                bads = config.bads[subject]
                raw.info['bads'] = bads
            print("no annotations found, reading bads from config")
            print("bads: ", raw.info['bads'])

        # rename / retype channels
        if config.set_channel_types is not None:
            raw.set_channel_types(config.set_channel_types)
        if config.rename_channels is not None:
            raw.rename_channels(config.rename_channels)

        # Fix coil types (does something only if needed). See:
        # https://martinos.org/mne/stable/generated/mne.channels.fix_mag_coil_types.html  # noqa
        raw.fix_mag_coil_types()

        if config.use_maxwell_filter:
            # To match their processing, transform to the head position of the
            # defined reference run
            extension = config.runs[config.mf_reference_run] + '_raw'
            refrun_in = op.join(meg_subject_dir,
                                config.base_fname.format(**locals()))

            info = mne.io.read_info(refrun_in)
            destination = info['dev_head_t']

            if config.mf_st_duration:
                print('    st_duration=%d' % (config.mf_st_duration,))

            raw_sss = mne.preprocessing.maxwell_filter(
                raw,
                calibration=config.mf_cal_fname,
                cross_talk=config.mf_ctc_fname,
                st_duration=config.mf_st_duration,
                origin=config.mf_head_origin,
                destination=destination)

            raw_sss.save(raw_fname_out, overwrite=True)

            if config.plot:
                # plot maxfiltered data
                raw_sss.plot(n_channels=50, butterfly=True,
                             group_by='position')

        else:

            raw.save(raw_fname_out, overwrite=True)
            print('')
            print('WARNING: The data were not maxfiltered. '
                  'They will be distorted')
            print('Set config.use_maxwell_filter=True '
                  'and run this script again')
            print('')

            if config.plot:
                # plot non-maxfiltered data
                raw.plot(n_channels=50, butterfly=True, group_by='position')

    n_raws += 1

    if n_raws == 0:
        raise ValueError('No input raw data found.')


parallel, run_func, _ = parallel_func(run_maxwell_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
