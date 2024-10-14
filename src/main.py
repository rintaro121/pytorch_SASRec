import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import model_selection, preprocessing

# from torch import concat, diag, logical_and, logical_or, nn, tensor, tile
from torch.utils.data import DataLoader

from dataset import MovielensDataset
from model import SASRec


class CFG:
    base_dir = "./datasets/ml-1m"
    ratings_path = os.path.join(base_dir, "ratings.dat")

    batch_size = 256
    max_len = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_size = 0.2
    seed = 42
    num_workers = os.cpu_count()

    num_epochs = 200
    hidden_units = 50
    num_heads = 1
    num_layers = 4
    dropout_rate = 0.2
    lr = 1e-3


if __name__ == "__main__":

    ratings_df = pd.read_csv(
        CFG.ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    ratings_df = ratings_df.groupby("movie_id").filter(lambda x: len(x) >= 5)

    le_movie = preprocessing.LabelEncoder()
    ratings_df["movie_id"] = le_movie.fit_transform(ratings_df.movie_id.values) + 1
    CFG.num_items = len(ratings_df["movie_id"].unique())

    seq_data_df = (
        ratings_df.sort_values(by="timestamp")
        .groupby("user_id")
        .agg(movie_list=("movie_id", list), movie_count=("movie_id", "size"))
        .reset_index()
    )
    seq_data_df = seq_data_df[seq_data_df["movie_count"] != 1]
    seq_data_df.reset_index()

    train_df, valid_df = model_selection.train_test_split(seq_data_df, test_size=CFG.test_size, random_state=CFG.seed)

    train_dataset = MovielensDataset(train_df, CFG.max_len, CFG.num_items, train_flag=True)
    valid_dataset = MovielensDataset(valid_df, CFG.max_len, CFG.num_items, train_flag=False)

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False)

    model = SASRec(
        CFG.num_items,
        CFG.hidden_units,
        CFG.max_len,
        CFG.num_heads,
        CFG.num_layers,
        CFG.dropout_rate,
        CFG.device,
    )
    model = model.to(CFG.device)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    valid_users = len(valid_dataset)

    for epoch in range(CFG.num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            inputs, pos_ids, neg_ids = batch

            pos_logits, neg_logits = model(inputs, pos_ids, neg_ids)
            pos_labels, neg_labels = torch.ones(pos_logits.shape), torch.zeros(neg_logits.shape)

            indices = np.where(pos_ids != 0)

            pos_logits = pos_logits.to(CFG.device)
            neg_logits = neg_logits.to(CFG.device)
            pos_labels = pos_labels.to(CFG.device)
            neg_labels = neg_labels.to(CFG.device)

            optimizer.zero_grad()

            loss = bce_loss(pos_logits[indices], pos_labels[indices])
            loss += bce_loss(neg_logits[indices], neg_labels[indices])

            for param in model.parameters():
                loss += 0.00005 * torch.norm(param)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Train [{epoch+1} / {CFG.num_epochs}] Loss : {running_loss}")

        with torch.no_grad():
            model.eval()
            hit_count = 0
            running_loss = 0.0
            for i, batch in enumerate(valid_dataloader):

                inputs, pos_ids, neg_ids, eval_item_ids = batch

                pos_logits, neg_logits = model(inputs, pos_ids, neg_ids)
                pos_labels, neg_labels = torch.ones(pos_logits.shape), torch.zeros(neg_logits.shape)

                indices = np.where(pos_ids != 0)

                pos_logits = pos_logits.to(CFG.device)
                neg_logits = neg_logits.to(CFG.device)
                pos_labels = pos_labels.to(CFG.device)
                neg_labels = neg_labels.to(CFG.device)

                loss = bce_loss(pos_logits[indices], pos_labels[indices])
                loss += bce_loss(neg_logits[indices], neg_labels[indices])

                running_loss += loss.item()

                eval_item_ids = eval_item_ids.to(CFG.device)
                last_item_ids = pos_ids[:, -1]

                output = model.predict(inputs)
                output = output[:, -1, :]

                eval_item_embs = model.item_emb(eval_item_ids)

                pred = eval_item_embs.matmul(output.unsqueeze(-1)).squeeze(-1)

                prob = F.softmax(pred, dim=-1)
                top_probabilities, top_indices = torch.topk(prob, k=10)
                HR = 0
                for i in range(inputs.size(0)):
                    if 0 in top_indices[i]:
                        HR += 1

        print(f"Valid [{epoch+1} / {CFG.num_epochs}] Loss : {running_loss}")
        print(f"HitCount:{HR}.  HitRate@10:{HR/valid_users}.")
        print()
