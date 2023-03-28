import numpy as np
import librosa

FRAME_LENGTH = 2048
HOP_LENGTH = 512
FRAME_SIZE = 2048
HOP_SIZE = 512


def amplitude_envelope(signal, frame_size=2048, hop_length=512):
    amplitude_envelope = []

    for i in range(0, len(signal), hop_length):
        current_frame_amplitude_envelope = max(signal[i:i + frame_size])
        amplitude_envelope.append(current_frame_amplitude_envelope)

    return np.array(amplitude_envelope)


def Rms(song):
    rms_song = librosa.feature.rms(y=song, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    return rms_song


def Zcr(song):
    zcr_song = librosa.feature.zero_crossing_rate(y=song, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    return zcr_song


def Mag_spec(song):
    signal_ft = np.fft.fft(song)
    magnitude_spectrum = np.abs(signal_ft)
    return magnitude_spectrum


def spectrogram(song):
    song_stft = librosa.stft(song, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    y_song = np.abs(song_stft) ** 2
    return y_song


def log_spec(song):
    spec_song = spectrogram(song)
    y_song_log = librosa.power_to_db(spec_song)


def log_mel(song, samp_rate):
    mel_spectrogram = librosa.feature.melspectrogram(y=song, n_fft=2048, sr=samp_rate, hop_length=512, n_mels=50)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram


def Mfcc(song, samp_rate, nmfcc=13):
    mfccs = librosa.feature.mfcc(y=song, n_mfcc=nmfcc, sr=samp_rate)
    return mfccs


def delta_mfcc(song, samp_rate, nmfcc=13):
    mfccs = Mfcc(song, samp_rate)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    delta3_mfcc = librosa.feature.delta(mfccs, order=3)
    delta4_mfcc = librosa.feature.delta(mfccs, order=4)
    delta5_mfcc = librosa.feature.delta(mfccs, order=5)
    delta6_mfcc = librosa.feature.delta(mfccs, order=6)

    return (delta_mfcc, delta2_mfcc, delta3_mfcc, delta4_mfcc, delta5_mfcc, delta6_mfcc)


def Chroma_stft(song, samp_rate):
    c_stft = librosa.feature.chroma_stft(y=song, sr=samp_rate)
    return c_stft


def Spec_centriod(song, samp_rate):
    return librosa.feature.spectral_centroid(y=song, sr=samp_rate)


# spectral rolloff is the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.
def spec_roll_off(song, samp_rate):
    return librosa.feature.spectral_rolloff(y=song, sr=samp_rate)


def spec_roll_off50(song, samp_rate):
    return librosa.feature.spectral_rolloff(y=song, sr=samp_rate, roll_percent=0.5)


def spec_roll_off25(song, samp_rate):
    return librosa.feature.spectral_rolloff(y=song, sr=samp_rate, roll_percent=0.25)


def spec_contrast(song, samp_rate):
    return librosa.feature.spectral_contrast(y=song, sr=samp_rate)


def perceptual_wt(song, samp_rate):
    return librosa.perceptual_weighting(S=song)


def tempogram(song, samp_rate):
    return librosa.feature.tempogram(y=song, sr=samp_rate)


def spec_bandwidth(song, samp_rate):
    return librosa.feature.spectral_bandwidth(y=song, sr=samp_rate)