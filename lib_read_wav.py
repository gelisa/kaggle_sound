import os
import pandas as pd
import struct
from scipy.io import wavfile as wav
import IPython.display as ipd
import matplotlib.pyplot as plt


def path_class(meta, wav_filename):
    excerpt = meta[meta.fname == wav_filename]
    path_name = os.path.join('audio_train',wav_filename)
    return path_name, excerpt['label'].values[0]


def read_one_file(filename):
    fs, data = wav.read(os.path.join('audio_train', filename))
    return fs, data


def wav_fmt_parser(meta, file_name):
    """
    Note a magic number 36 in many places of the code. This is struct.calcsize(wave_header_format)
    :param meta:
    :param file_name:
    :return:
    """
    full_path, class_label = path_class(meta, file_name)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    rate, wav_sample = wav.read(full_path)
    #print(riff_fmt)
    n_channels_string = riff_fmt[22:24]
    n_channels = struct.unpack("H",n_channels_string)[0]
    s_rate_string = riff_fmt[24:28]
    s_rate = struct.unpack("I",s_rate_string)[0]
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    return n_channels, s_rate, bit_depth, len(wav_sample)


def wav_plotter(meta, file_name):
    full_path, class_label = path_class(meta, file_name)
    n_channels, s_rate, bit_depth, n_samples = wav_fmt_parser(meta, file_name)
    rate, wav_sample = wav.read(full_path)
    print(file_name)
    print('Label: {}'.format(class_label))
    print('Number of channels: {}'.format(n_channels))
    print('Sampling rate: {}'.format(s_rate))
    print('Bit depth: {}'.format(bit_depth))
    f, (ax0, ax1) = plt.subplots(figsize=(12, 8), nrows=2)
    ax0.plot(wav_sample)
    ax0.set_xlabel('sample')
    ax0.set_ylabel('amplitude')
    ax1.specgram(wav_sample, Fs=44100)
    ax1.set_xlabel('time')
    ax1.set_ylabel('frequency')
    return ipd.Audio(full_path)


def read_fmt_data(meta):
    return pd.DataFrame(
        meta.fname.apply(lambda x: wav_fmt_parser(meta, x)).tolist(),
        columns=['n_chan', 's_freq', 'bit_depth', 'n_samples'], index=meta.index
    )