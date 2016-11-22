from collections import defaultdict

import pickle
from speech_recognition import Recognizer

from speaker_recognition.exceptions import FeatureExtractionException
from speaker_recognition.feature import mix_feature
from speaker_recognition.gmmset import GMMSetPyGMM as GMMSet


class SpeakerRecognizer(Recognizer):
    def __init__(self):
        super(SpeakerRecognizer, self).__init__()
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    def enroll(self, audio_data, label):
        feat = mix_feature(audio_data)
        self.features[label].extend(feat)

    def predict(self, audio_data):
        try:
            feat = mix_feature(audio_data)
        except FeatureExtractionException:
            return None
        return self.gmmset.predict_one(feat)

    def train(self):
        for name, feats in self.features.iteritems():
            self.gmmset.fit_new(feats, name)

    def dump(self, fn):
        """ dump all models to file"""
        self.gmmset.before_pickle()
        with open(fn, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    def dumps(self):
        self.gmmset.before_pickle()
        s = pickle.dumps(self, -1)
        self.gmmset.after_pickle()
        return s

    @staticmethod
    def load(fn):
        """ load from a dumped model file"""
        with open(fn, 'rb') as f:
            R = pickle.load(f)
        R.gmmset.after_pickle()
        return R

    @staticmethod
    def loads(string):
        R = pickle.loads(string)
        R.gmmset.after_pickle()
        return R
