import torch
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

def align_audio(y, target_len):
    """
        Align audio lengths to specified values, crop the longer and pad the shorter.
        :param y: audio, torch tensor, shape: (channel_num, sample_len)
        :param target_len:
        :return: aligned audio.
        :example: ~(y, 20000), where y has shape (1, 16000), return shape(1, 20000)
    """
    l = y.shape[-1]
    # # 固定裁切居中部分，保证过程确定性
    if l > target_len:
        start_idx = (l - target_len) // 2
        y = y[:, start_idx: start_idx + target_len]

    # 两端补零
    if l < target_len:
        pad_len = target_len - l
        pad_left = pad_len // 2
        y = torch.nn.functional.pad(y, (pad_left, pad_len - pad_left))

    return y