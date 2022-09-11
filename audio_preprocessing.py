import os
import numpy as np
import librosa

class AudioPreprocessor():
    '''
    Preprocessing pipeline for audio files.

    - Load files with load()
    - Fit audio signals to desired duration with pad()
    - Extract spectrograms with log_spectrogram()
    - Save data with save_data()

    Attributes:
        - signals
        - spectrograms
    '''

    def __init__(self):

        self.signals = None
        self.spectrograms = None

        self.sample_rate = None
        self.signal_length = None
        self.frame_size = None
        self.hop_length = None

    def load(self,path,sample_rate=22050):
        '''
        Load audio files from folder.

        :param path: folder path for audio files.
        :param sample_rate: sample rate.
        :return: None
        '''

        self.sample_rate = sample_rate
        self.signals = [librosa.load(os.path.join(path, file), sr=sample_rate)[0]
                        for file in os.listdir(path) if not file.startswith('.')]

    def pad(self,duration=.74):
        '''
        Trim or pad audio signals to uniform duration. Uses left padding and trims from start to set duration.

        :param duration: duration in seconds.
        :return: None
        '''

        self.signal_length = int(self.sample_rate*duration)

        signal_list = []

        for signal in self.signals:
            if signal.shape[0] < self.signal_length:
                signal = np.pad(signal, (self.signal_length - signal.shape[0], 0))
            else:
                signal = signal[:self.signal_length]
            signal_list.append(signal)

        self.signals = np.array(signal_list)

    def log_spectrogram(self,frame_size=512,hop_length=256):
        '''
        Uses short-time fourier transform to extract spectrogram,
        then transforms amplitude data to log-scale.

        :param frame_size: frame size
        :param hop_length: hop length
        :return: None
        '''

        self.frame_size = frame_size
        self.hop_length = hop_length
        spectrogram_list = []

        for signal in self.signals:
            stft = librosa.stft(signal,n_fft=self.frame_size,hop_length=self.hop_length)[:-1]
            log_spectrogram = librosa.amplitude_to_db(np.abs(stft))
            spectrogram_list.append(log_spectrogram)

        self.spectrograms = np.array(spectrogram_list)

    def normalize(self):
        ''' Normalizes spectrograms to a maximum value of 1 and minimum of 0.'''

        norm_spectrogram_list = []
        for spectrogram in self.spectrograms:
            norm_spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
            norm_spectrogram_list.append(norm_spectrogram)

        self.spectrograms = np.array(norm_spectrogram_list)

    def save_data(self,path):
        ''' Saves signal and spectrogram arrays to designated path.'''
        os.mkdir(path)
        np.save(os.path.join(path,'signals.npy'), self.signals)
        np.save(os.path.join(path,'spectrograms.npy'), self.spectrograms)