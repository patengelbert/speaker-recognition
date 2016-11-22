import BOB as MFCC
import LPC
import numpy as np

from speaker_recognition.exceptions import FeatureExtractionException


def mix_feature(audio_data):
    data = audio_data.get_raw_data(convert_width=2) # Convert to a 16 bit int value
    signal = np.cast[np.float64](np.fromstring(data, np.int16))
    mfcc = MFCC.extract(audio_data.sample_rate, signal)
    lpc = LPC.extract(audio_data.sample_rate, signal)

    if len(mfcc) == 0:
        raise FeatureExtractionException()

    return np.concatenate((mfcc, lpc), axis=1)
