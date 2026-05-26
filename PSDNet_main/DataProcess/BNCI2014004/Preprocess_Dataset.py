from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery
import numpy as np
import os
import sys
# Getcurrentfile所在directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# 将项目根directory添加到Pythonpath
sys.path.append(project_root)
print(f"currentdirectory: {current_dir}")
print(f"项目根directory: {project_root}")

# 现在可以导入utils模块中的类和函数
from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl

def download_data_BNCI2014004():
    dataset = BNCI2014_004()
    datasetname = 'BNCI2014004'

    # use MotorImagery paradigm，Setpreprocess参数
    paradigm = MotorImagery(fmin=0, fmax=120)
    alldata = paradigm.get_data(dataset)
    data, labels_string, metadata = alldata
    sessions = metadata["session"].values
    
    # 只保留session为0train的data
    train_mask = sessions == '3test' 
    data = data[train_mask, :, :]
    labels_string = [labels_string[i] for i in np.where(train_mask)[0]]
    metadata = metadata[train_mask]
    
    # Getsubject信息
    subjects = metadata["subject"].values
    
    # channelsname和采样率
    ch_names = ["C3",  "CZ",  "C4"]
    sampling_rate = 250
    
    # labels映射
    label_map = {
        "left_hand": 0,
        "right_hand": 1,
    }
    
    # 将labelsconvert为数字
    labels = np.array([label_map[label] for label in labels_string])
    
    # CreateEEGData实例
    eeg_data_obj = EEGData(
        dataset_name=datasetname,
        eeg_data=data,
        subject_ids=subjects,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        labels=labels
    )
    
    # Getcurrentfile所在directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # use相对pathSavedatafile到同级directory
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=current_dir)
    
    return eeg_data_obj


if __name__ == "__main__":
    # test代码可行性
    try:
        print("开始testPreprocess_Dataset.py...")
        # 下载data
        eeg_data_obj = download_data_BNCI2014004()
        print("data下载完成")

        file_path = os.path.join(current_dir, 'BNCI2014004.pkl')
        print(f"尝试从fileLoaddata: {file_path}")
        eeg_data = load_eeg_data_from_pkl(file_path)
        
        # Print返回对象的type
        print(f"\n函数返回type: {type(eeg_data)}")
        
        # 验证返回的是EEGData对象
        if isinstance(eeg_data, EEGData):
            print("验证success: 函数返回了EEGData对象")
            
            # Printdata集的基本信息
            print("\n=== data集基本信息 ===")
            print(str(eeg_data))
            
            # Checkdatashape
            print(f"\ndatashape: {eeg_data.eeg_data.shape}")
            print(f"subjectIDcount: {eeg_data.subject_ids.shape[0]}")
            print(f"channelscount: {len(eeg_data.channel_names)}")
            # Printchannelsname（一行显示，用顿号间隔）
            print(f"channelsname列表: {'、'.join([f'{i+1}. {channel}' for i, channel in enumerate(eeg_data.channel_names)])}")
            
            # Checklabels信息
            if eeg_data.labels is not None:
                print(f"labelscount: {eeg_data.labels.shape[0]}")
                unique_labels, counts = np.unique(eeg_data.labels, return_counts=True)
                print("labels分布:")
                for label, count in zip(unique_labels, counts):
                    print(f"  labels {label}: {count} 个samples")
        else:
            print("验证fail: 函数没有返回EEGData对象")
            
        print("\ntest完成!")
        
    except Exception as e:
        print(f"test过程中出现error: {str(e)}")
        import traceback
        traceback.print_exc()