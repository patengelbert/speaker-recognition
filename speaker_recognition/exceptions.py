class SpeakerRecogniserException(Exception):
    def __init__(self, err=None):
        self.err = err

    def __str__(self):
        return str(self.err)


class FeatureExtractionException(SpeakerRecogniserException):
    def __init__(self, err="Unable to extract feature"):
        super(FeatureExtractionException, self).__init__(err)
