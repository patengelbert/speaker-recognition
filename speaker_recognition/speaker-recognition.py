#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: speaker-recognition.py
# Date: Sun Feb 22 22:36:46 2015 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import glob
import itertools
import os
import sys

from speech_recognition import AudioFile

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from speaker_recognition.recogniser import SpeakerRecognizer

r = SpeakerRecognizer()


def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.

Examples:
    Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out

    Predict (predict the speaker of all wav files):
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                        help='Task to do. Either "enroll" or "predict"',
                        required=True)

    parser.add_argument('-i', '--input',
                        help='Input Files(to predict) or Directories(to enroll)',
                        required=True)

    parser.add_argument('-m', '--model',
                        help='Model file to save(in enroll) or use(in predict)',
                        required=True)

    ret = parser.parse_args()
    return ret


def task_enroll(input_dirs, output_model):
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = []
    if len(dirs) == 0:
        print "No valid directory found!"
        sys.exit(1)
    for d in dirs:
        label = os.path.basename(d.rstrip('/'))

        wavs = glob.glob(d + '/*.wav')
        if len(wavs) == 0:
            print "No wav file found in {0}".format(d)
            continue
        print "Label {0} has files {1}".format(label, ','.join(wavs))
        for wav in wavs:
            with AudioFile(wav) as file:
                data = r.record(file)
            r.enroll(data, label)

    r.train()
    r.dump(output_model)


def task_predict(input_files, input_model):
    r = SpeakerRecognizer.load(input_model)
    for f in glob.glob(os.path.expanduser(input_files)):
        with AudioFile(f) as file:
            data = r.record(file)
        label = r.predict(data)
        print f, '->', label


if __name__ == '__main__':
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)
