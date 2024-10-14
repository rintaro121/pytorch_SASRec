import random

import torch
from torch.utils.data import Dataset


class MovielensDataset(Dataset):
    def __init__(self, df, max_len, num_items, train_flag):
        self.df = df
        self.max_len = max_len
        self.num_items = num_items
        self.seq_data = []
        self.train_flag = train_flag

        self.item_set = set(range(1, num_items + 1))

        for i in range(len(df)):
            row = self.df.iloc[i]
            # idxes = self.random_idxes(row.movie_count)

            movie_sequence = row.movie_list[:-1]
            input_ids, labels = self.padding_sequence(movie_sequence)
            # input_ids = torch.Tensor(self.padding_sequence(movie_seq, self.max_length)).long()
            # label = torch.Tensor(row.movie_list[:1])

            self.seq_data.append(
                {
                    "orig_movie_sequence": movie_sequence,
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        data = self.seq_data[idx]

        orig_movie_sequence = data["orig_movie_sequence"]
        input_ids = data["input_ids"]
        labels = data["labels"]
        negatives = self.negative_sampling(orig_movie_sequence, input_ids)

        if self.train_flag:
            return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(negatives)
        else:
            last_item_id = labels[-1]
            negative_set = self.item_set - set(orig_movie_sequence)
            negative_indices = random.sample(list(negative_set), 100)

            # 101 items in total (1 positive item, 100 negative items)
            eval_item_ids = [last_item_id] + negative_indices

            return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(negatives), torch.tensor(eval_item_ids)

    def padding_sequence(self, orig_movie_sequence):
        # negative_set = self.item_set - set(sequence)

        sequence = orig_movie_sequence[-(self.max_len + 1) :]
        inputs = sequence[:-1]
        labels = sequence[1:]

        seq_len = min(len(sequence) - 1, self.max_len)

        inputs = (self.max_len - seq_len) * [0] + inputs
        labels = (self.max_len - seq_len) * [0] + labels
        return inputs, labels

    def negative_sampling(self, orig_movie_sequence, input_ids):
        negative_set = self.item_set - set(orig_movie_sequence)
        sequence = orig_movie_sequence[-(self.max_len + 1) :]
        seq_len = min(len(sequence) - 1, self.max_len)

        negatives = (self.max_len - seq_len) * [0] + random.sample(list(negative_set), seq_len)
        return negatives
