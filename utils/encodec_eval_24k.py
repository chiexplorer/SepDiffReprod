from encodec import EncodecModel
from encodec.utils import convert_audio
from encodec.compress import compress, decompress, MODELS
from encodec.utils import save_audio, convert_audio

import torchaudio
import torch
from pathlib import Path
import os, time


if __name__ == '__main__':
    fpath = r"H:\exp\output\ground_truth\s1\1089-134686-0000_61-70968-0047.wav"
    savepath = Path(os.path.join(r"D:\Projects\pyprog\SepDiffReprod\files", os.path.basename(fpath)))
    save_sr = 24000
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
    model.set_target_bandwidth(6.0)

    """
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(fpath)
    print("audio duration & length: ", wav.shape[-1] / float(sr), wav.shape[-1])
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    audio_length = wav.shape[-1]
    print("resample length: ", audio_length)
    # # 压缩&解压，但不经过ecdc化
    wav = wav.unsqueeze(0)
    """
    wav = torch.randn(2, 1, 82000)
    audio_length = wav.shape[-1]
    # 编码&量化
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1) # [B, n_q, T]
    print("codes shape: ", codes.shape)
    frame = [(codes, None), ]  # 按decode要求的输入形式封装
    # 解码
    with torch.no_grad():
        wav_recon = model.decode(frame)
    wav_recon = wav_recon[:, :, :audio_length]  # 长度对齐输入
    print("recon shape: ", wav_recon.shape)
    print("recon duration: ", wav_recon.shape[-1] / float(save_sr))
    save_audio(wav_recon, savepath, save_sr, rescale=False)

    # # # 压缩，然后立即解压缩
    # start_time = time.time()
    # compressed = compress(model, wav, use_lm=False)
    # encode_time = time.time()
    # print("encode time: ", encode_time - start_time)
    # out, out_sample_rate = decompress(compressed)
    # print("decode time: ", time.time() - encode_time)
    # save_audio(out, savepath, out_sample_rate, rescale=False)
