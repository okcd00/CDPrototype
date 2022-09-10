import os
import time
from glob import glob
from copy import deepcopy
from torch.utils.data import Dataset
from utils import load_json, dump_json, lower_bound, highlight_positions
from utils.text_utils import clean_text


class PureTextDataset(Dataset):
    def __init__(self, fp):
        self.fp = fp
        self.file_list = sorted(glob(f"{fp}/*.txt"))
        self.file_sample_count = []
        self.file_offset = [0]
        self.sample_counts = self.count_samples()
        self.current_file_index = -1
        self.current_file_samples = []
        print(f"Loaded {self.file_list.__len__()} files from {fp}.")

    @staticmethod
    def read_text_file(path):
        return [line.strip() for line in open(path, 'r') if line.strip()]

    def remove_dataset_info(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        if os.path.exists(fp_log_path):
            os.remove(fp_log_path)

    def dump_dataset_info(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        dump_json({
            'file_offset': self.file_offset,
            'file_sample_count': self.file_sample_count,
            'sample_counts': self.sample_counts,
        }, fp_log_path)

    def reset_dataset_info(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        self.remove_dataset_info()
        self.count_samples()

    def count_samples(self):
        fp_log_path = f"{self.fp}/dataset_info.log"
        start_time = time.time()
        if os.path.exists(fp_log_path):
            dataset_info = load_json(fp_log_path)
            self.file_offset = list(map(int, dataset_info['file_offset']))
            self.file_sample_count = dataset_info['file_sample_count']
            self.sample_counts = dataset_info['sample_counts']
        else:
            for file_name in self.file_list:
                samples = self.read_text_file(file_name)
                s_len = len(samples)
                self.file_sample_count.append(s_len)
                self.file_offset.append(self.file_offset[-1] + s_len)
            self.sample_counts = sum(self.file_sample_count)
            self.dump_dataset_info()
        print(f"Init indexing ends in {time.time()-start_time} seconds")
        return self.sample_counts

    def load_from_dir(self, dir_path):
        self.__init__(dir_path)

    @staticmethod
    def binary_search_file_index(a, x):
        # the index of the file for x-th sample.
        return lower_bound(a, x + 1) - 1

    def show_item(self, tup):
        if isinstance(tup, int):
            tup = self[tup]
        if isinstance(tup, list):
            highlight_positions(
                text=tup[0], positions=tup[2], color='blue')
            highlight_positions(
                text=tup[1], positions=tup[2], color='blue')
        elif isinstance(tup, dict):
            highlight_positions(
                text=tup['original_text'], positions=tup['wrong_ids'], color='blue')
            highlight_positions(
                text=tup['correct_text'], positions=tup['wrong_ids'], color='blue')

    def __len__(self):
        return self.sample_counts

    def __getitem__(self, index):
        # for a large text corpus, shuffle is not recommended.
        file_index = self.binary_search_file_index(self.file_offset, index)
        if file_index == self.file_list.__len__():
            raise ValueError(f"Invalid index {file_index} with offset {index}")
        if file_index != self.current_file_index:
            file_path = self.file_list[file_index]
            self.current_file_samples = self.read_text_file(file_path)
            self.current_file_index = file_index
        index_in_file = index - self.file_offset[file_index]
        target_text = clean_text(deepcopy(self.current_file_samples[index_in_file]))[:500]
        # return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']
        return target_text, target_text, []


if __name__ == "__main__":
    ptd = PureTextDataset("/data/chendian/clean_pretrain_data/")
    print(ptd.sample_counts)
