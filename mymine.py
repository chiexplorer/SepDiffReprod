import argparse, os, json
import torch

def train(config):
    pass


if __name__ == '__main__':

    print('Initializing Inference Process..')

    # 载入配置文件
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_wavs_dir', default='test_files')
    # parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', default="")
    parser.add_argument('--config', default='')

    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    print(json_config)

    if torch.cuda.is_available():
        print("train with gpu")
        pass
    else:
        print("check your gpu")