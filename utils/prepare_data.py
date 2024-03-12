import csv, os
import librosa

"""
    准备数据加载所需的csv文件（记录每条数据包含的文件路径）
"""
def create_aishell1mix2_csv(
        datapath,
        savepath,
        addnoise=False,
        version="",
        set_types=["train", "val", "test"],
):
    """
    This functions creates the .csv file for the aishell1mix2 dataset
    @:param datapath: parant dir of 'version'
    @:param savepath: output dir
    @:param version: common prefix of directory to wavs
    @:param set_types: dirname to divide dataset
    """

    for set_type in set_types:
        # 是否使用加噪版本
        if addnoise:
            mix_path = os.path.join(datapath, version, set_type, "mix_both/")
        else:
            mix_path = os.path.join(datapath, version, set_type, "mix_clean/")

        s1_path = os.path.join(datapath, version, set_type, "s1/")
        s2_path = os.path.join(datapath, version, set_type, "s2/")
        noise_path = os.path.join(datapath, version, set_type, "noise/")

        files = os.listdir(mix_path)

        # 子集中的所有音频路径
        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "s1_wav",
            "s2_wav",
            "noise_wav",
        ]

        with open(
                savepath + "/aishell1mix2_" + set_type + ".csv", "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, noise_path) in enumerate(
                    zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, noise_fl_paths)
            ):
                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "s1_wav": s1_path,
                    "s2_wav": s2_path,
                    "noise_wav": noise_path,
                }
                writer.writerow(row)

def create_libri2mix_csv(
    datapath,
    savepath,
    addnoise=False,
    version="wav8k/min/",
    set_types=["train-360", "dev", "test"],
):
    """
    This functions creates the .csv file for the libri2mix dataset
    """

    for set_type in set_types:
        if addnoise:
            mix_path = os.path.join(datapath, version, set_type, "mix_both/")
        else:
            mix_path = os.path.join(datapath, version, set_type, "mix_clean/")

        s1_path = os.path.join(datapath, version, set_type, "s1/")
        s2_path = os.path.join(datapath, version, set_type, "s2/")
        noise_path = os.path.join(datapath, version, set_type, "noise/")

        files = os.listdir(mix_path)

        mix_fl_paths = [mix_path + fl for fl in files]
        s1_fl_paths = [s1_path + fl for fl in files]
        s2_fl_paths = [s2_path + fl for fl in files]
        noise_fl_paths = [noise_path + fl for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(savepath + "/libri2mix_" + set_type + ".csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path, noise_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, noise_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                    "noise_wav": noise_path,
                    "noise_wav_format": "wav",
                    "noise_wav_opts": None,
                }
                writer.writerow(row)


if __name__ == '__main__':
    # fpath = r"D:\Data\AiShell1-mix-test"
    # fpath = r"D:\Data\LibriMixAligned\Libri2Mix"
    fpath = r"H:\exp\dataset\LibriMixAligned\Libri2Mix"  # 专用固态

    savepath = "../save/LibriMixAligned"
    # create_aishell1mix2_csv(datapath=fpath, savepath=savepath)
    create_libri2mix_csv(
        datapath=fpath,
        savepath=savepath,
        version="wav16k/min/",
        set_types=["train-100", "test", "dev"]
    )