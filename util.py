import os
import numpy as np
import pandas as pd
import json
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class DataPreprocess:
    def __init__(self, separate_unit, file_name):
        self.path = file_name
        self.file_list = os.listdir("./data")
        if self.path not in self.file_list:
            raise ValueError("File Name Error, Check Your CSV Data File Name")
        self.df = pd.read_csv("./data/" + self.path)
        self.attributes = self.df.columns
        self.data = self.separating(self.df, separate_unit)
        self.seq_len = len(self.data[0]["df"])
        self.att_len = len(self.attributes)
        self.main_forward(self.data)

    def separating(self, df, separate_unit):
        df_data = []
        temp_dict = {}
        print("Data Separating...")
        if isinstance(separate_unit, list):
            for unit in separate_unit:
                try:
                    temp = df[df[self.attributes[0]].str.contains(unit)]
                except:
                    raise ValueError("Wrong separte_unit, check your data is it can separate")
                mean = []
                std = []
                for att in self.attributes:
                    if att == self.attributes[0]:
                        continue
                    mean.append(temp[att].mean())
                    std.append(temp[att].std())
                temp_dict["df"] = temp.drop("time", axis=1)
                temp_dict["mean"] = mean
                temp_dict["std"] = std
                df_data.append(temp_dict)
            return df_data
        elif isinstance(separate_unit, int):
            for i in range(len(df) // separate_unit):
                temp = df[separate_unit*i:separate_unit*(i+1)]
                mean = []
                std = []
                for att in self.attributes:
                    if att == self.attributes[0]:
                        continue
                    mean.append(temp[att].mean())
                    std.append(temp[att].std())
                temp_dict["df"] = temp.drop("time", axis=1)
                temp_dict["mean"] = mean
                temp_dict["std"] = std
                df_data.append(temp_dict)
            return df_data

    def parse_delta(self, masks, dir_):
        if dir_ == 'backward':
            masks = masks[::-1]
        deltas = []
        for h in range(self.seq_len):
            if h == 0:
                deltas.append(np.ones(len(self.attributes.tolist()) - 1))
            else:
                deltas.append(np.ones(len(self.attributes.tolist()) - 1) + (1 - masks[h]) * deltas[-1])
        return np.array(deltas)

    def parse_rec(self, values, masks, evals, eval_masks, dir_):
        deltas = self.parse_delta(masks, dir_)
        forwards = pd.DataFrame(values).fillna(method='ffill').fillna(method="bfill").values
        rec = {}
        rec['values'] = np.nan_to_num(values).tolist()
        rec['masks'] = masks.astype('int32').tolist()
        rec['evals'] = np.nan_to_num(evals).tolist()
        rec['eval_masks'] = eval_masks.astype('int32').tolist()
        rec['forwards'] = forwards.tolist()
        rec['deltas'] = deltas.tolist()
        return rec

    def parse_file(self, df_data, open_file, data_num):
        evals = df_data["df"].to_numpy()
        evals = (np.array(evals) - df_data["mean"]) / df_data["std"]
        shp = evals.shape
        evals = evals.reshape(-1)
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, len(indices) // 10)
        values = evals.copy()
        values[indices] = np.nan
        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))
        evals = evals.reshape(shp)
        values = values.reshape(shp)
        masks = masks.reshape(shp)
        eval_masks = eval_masks.reshape(shp)
        rec = {'label': data_num}
        rec['forward'] = self.parse_rec(values, masks, evals, eval_masks, dir_='forward')
        rec['backward'] = self.parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
        rec = json.dumps(rec)
        open_file.write(rec + '\n')

    def main_forward(self, data_list):
        with open("json/processed_data.json", "w") as fs:
            n = 0
            print("Data Setting...")
            for data in data_list:
                try:
                    self.parse_file(data, fs, n)
                    n += 1
                except Exception as e:
                    print(e)
                    continue

    def need_parameters(self):
        return self.seq_len, self.att_len


class BRITSDataset(Dataset):
    def __init__(self):
        super(BRITSDataset, self).__init__()
        self.content = open('./json/processed_data.json').readlines()
        indices = np.arange(len(self.content))
        self.val_indices = np.random.choice(indices, len(self.content) // 5, replace=False).tolist()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))
        # ground truth
        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))
        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}
    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))
    return ret_dict


def get_loader(batch_size=64, shuffle=True, num_workers=0):
    print("Setting Batch...")
    data_set = BRITSDataset()
    data_iter = DataLoader(dataset=data_set,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           pin_memory=True,
                           collate_fn=collate_fn)
    return data_iter


def to_var(var):
    if torch.is_tensor(var):
        var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var