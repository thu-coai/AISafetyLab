import json
import os
from loguru import logger
from .example import Example
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset

class AttackDataset:
    def __init__(self, path, subset_slice=None):
        if isinstance(path, list):
            self.data = path
        elif os.path.exists(path):
            # logger.warning('You chose to load a dataset from a local file. If it is not a json/jsonl file, please make sure you correctly set \'local_file_type\' parameter when you instantiate the AttackDataset.')

            if path.endswith('.json'):
                with open(path) as f:
                    _data = json.load(f)
            elif path.endswith('.jsonl'):
                _data = pd.read_json(path, lines=True).to_dict('records')
            elif path.endswith('.csv'):
                _data = pd.read_csv(path).to_dict('records')
            elif path.endswith('.xlsx'):
                _data = pd.read_excel(path).to_dict('records')
            elif path.endswith('.parquet'):
                _data = pd.read_parquet(path).to_dict('records')

            self.data = [Example(**d) for d in _data]

        

        # ... 你的其他代码 ...

        elif 'thu-coai/AISafetyLab_Datasets' in path:
            # 1. 定义你本地数据集文件夹的路径
            #    请根据你的实际情况修改这个路径
            local_data_dir = "/data3/zhangqinglin/project/AISafetyLab/AISafetyLab_Datasets"

            # 2. 从输入路径中解析出不带扩展名的文件名
            #    这部分保留了原始代码的逻辑，例如 path 可能为 ".../advbench"
            file_name_without_ext = path.split('/')[-1]

            # 3. 构建本地数据文件的完整路径 (我们假设文件都是 .json 格式)
            local_file_path = os.path.join(local_data_dir, file_name_without_ext + ".json")

            # (可选但推荐) 检查文件是否存在，如果不存在则报错
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(
                    f"错误：找不到本地数据文件 '{local_file_path}'\n"
                    f"请检查 'local_data_dir' 变量是否设置正确，以及该文件是否存在。"
                )
            
            # 4. 从本地 JSON 文件加载数据
            #    对于单个JSON文件，datasets库通常会将其加载为默认的 'train' split
            _data = load_dataset('json', data_files=local_file_path, split='train')
            
            self.data = [Example(**d) for d in _data]

# ... 你的其他代码 ...
        else:
            logger.error('Please make sure the path is a valid local file path or a huggingface dataset path. For local files, we support the following types: json, jsonl, csv, xlsx, parquet. For huggingface datasets, we now only support the datasets in the "thu-coai/AISafetyLab_Datasets" repository. Please open an issue if you want to use other datasets.')
            raise NotImplementedError(
                'Please make sure the path is a valid local file path or a huggingface dataset path. For local files, we support the following types: json, jsonl, csv, xlsx, parquet. For huggingface datasets, we now only support the datasets in the "thu-coai/AISafetyLab_Datasets" repository. Please open an issue if you want to use other datasets.'
            )

        # logger.info(f'subset slice:{subset_slice}')
        if subset_slice is not None:
            if isinstance(subset_slice, int):
                self.data = self.data[:subset_slice]
            elif isinstance(subset_slice, list):
                _slice = slice(*subset_slice)
                self.data = self.data[_slice]
            else:
                self.data = self.data[subset_slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AugmentDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, augment_target, shuffle):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.effective_dataset_size = len(self.dataset)
        self.aufgment_target = augment_target

        self.process_fn = lambda s: s.replace("Sure, here is", "Sure, here's")
        self.process_fn2 = lambda s: s.replace("Sure, h", "H")

    def __iter__(self):
        for batch in super(AugmentDataLoader, self).__iter__():
            if self.aufgment_target:
                targets = []
                for target in batch["target"]:
                    if np.random.random() < 0.5:
                        target = self.process_fn(target)
                    if np.random.random() < 0.5:
                        target = self.process_fn2(target)
                    targets.append(target)
                batch["target"] = targets
            yield batch

def get_dataloader(data_pth, batch_size, shuffle, augment_target):
    dataset = AttackDataset(path=data_pth)
    dataloader = AugmentDataLoader(
        dataset, augment_target=augment_target, shuffle=shuffle, batch_size=batch_size
    )
    return dataloader
