from utils.meldataset import mel_spectrogram, MAX_WAV_VALUE

def get_mel(x, h):
    """
    Get mel-spectrogram
    :param x: speech signal, torch tensor
    :param h: hparams dict
    :return: mel spec
    """
    return mel_spectrogram(x, h['n_fft'], h['num_mels'],
                           h['sampling_rate'], h['hop_size'],
                           h['win_size'], h['fmin'], h['fmax'])