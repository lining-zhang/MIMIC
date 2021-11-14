import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import defaultdict
from transformer_encoder import Encoder
from data_dealer import read_txt_file, Tokenizer, SequenceDataset, pad_collate_fn, mask_tokens


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, config):
        self.config = config

        self.data_path = {"train": self.config["train_path"],
                          "valid": self.config["valid_path"],
                          "test": self.config["test_path"]}
        self.dataset = {"train": self.create_dataset(self.data_path["train"])[0],
                        "valid": self.create_dataset(self.data_path["valid"])[0],
                        "test": self.create_dataset(self.data_path["test"])[0]}
        self.dataloader = {"train": self.create_dataloader(self.dataset["train"]),
                           "valid": self.create_dataloader(self.dataset["valid"]),
                           "test": self.create_dataloader(self.dataset["test"])}
        self.max_len = np.max([self.create_dataset(self.data_path["train"])[2],
                               self.create_dataset(self.data_path["valid"])[2],
                               self.create_dataset(self.data_path["test"])[2]])

        self.vocab_obj_path = self.config["vocab_obj_path"]
        self.dictionary_obj = self.create_dataset(self.data_path["train"])[1]
        self.model_path = self.config["model_path"]

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        print('Is GPU available: {}'.format(torch.cuda.is_available()))
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)

    def create_dataset(self, data_path):
        notes = read_txt_file(data_path)
        tokenizer_obj = Tokenizer(notes, self.vocab_obj_path)
        tokenized_ds = tokenizer_obj.tokenized_ds
        max_len = np.max([len(t) for t in tokenized_ds])
        return SequenceDataset(tokenized_ds), tokenizer_obj.dictionary_obj, max_len

    def create_dataloader(self, dataset):
        dataloader = DataLoader(dataset,
                                batch_size=self.config["batch_size"],
                                collate_fn=lambda x: pad_collate_fn(self.dictionary_obj.get_id('<pad>'), x),
                                drop_last=True,
                                shuffle=False)
        return dataloader

    def load_model(self, model, model_name=None):
        if self.config["is_train"]:
            model = model.to(self.device)
        else:
            model.load_state_dict(torch.load(os.path.join(self.model_path, model_name), map_location=self.device))
            model.to(self.device)
        return model

    def train(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        model.train()
        for epoch in range(self.config["num_epoch"]):
            epoch = epoch + 1
            print('Train Epoch: {}'.format(epoch))
            losses = 0

            for iteration, batch in enumerate(self.dataloader["train"]):
                iteration = iteration + 1
                # Mask the batch
                inputs, labels = mask_tokens(batch,
                                             mask_prob=self.config["mask_prob"],
                                             pad_token_id=self.dictionary_obj.get_id('<pad>'),
                                             mask_token_id=self.dictionary_obj.get_id('[M]'),
                                             vsize=len(self.dictionary_obj))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # forward
                outputs, _ = model(inputs)
                outputs_ = outputs.view(-1, outputs.size(-1))
                labels_ = labels.reshape(-1)
                loss = self.criterion(outputs_, labels_)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                if iteration == 2000:
                    print("Training Loss: %.8f" % (loss.item()))

            avg_loss = losses / len(self.dataloader["train"])
            print("Epoch %d \tAvg Train Loss: %.8f" % (epoch, avg_loss))

            torch.save(model.state_dict(),
                       os.path.join(self.config["model_path"], 'transformer_encoder_epoch{}.pt'.format(epoch)))

            self.eval(model, "valid")

    def eval(self, model, dataloader_type):
        model.eval()
        losses = 0
        stats = defaultdict(list)

        for batch in self.dataloader[dataloader_type]:
            with torch.no_grad():
                inputs, labels = mask_tokens(batch,
                                             mask_prob=self.config["mask_prob"],
                                             pad_token_id=self.dictionary_obj.get_id('<pad>'),
                                             mask_token_id=self.dictionary_obj.get_id('[M]'),
                                             vsize=len(self.dictionary_obj))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs, attentions = model(inputs)
                outputs_ = outputs.view(-1, outputs.size(-1))
                labels_ = labels.reshape(-1)
                loss = self.criterion(outputs_, labels_)

                losses += loss.item()
                stats['attentions'].append(attentions)

        avg_loss = losses / len(self.dataloader[dataloader_type])
        print("===== {} loss: %.3f =====".format(dataloader_type) % (avg_loss))

        # save attentions
        if dataloader_type == "test":
            with open(os.path.join(self.config["attn_path"], 'attentions.pkl'), 'wb') as fp:
                pickle.dump(stats['attentions'], fp)


class Transformer_Encoder_Model(nn.Module):
    def __init__(self,
                 dim,
                 max_len,
                 vocab_size,
                 pad_index,
                 num_layers,
                 ffn_hidden,
                 n_head,
                 drop_prob,
                 eps):
        super().__init__()
        self.encoder = Encoder(dim, max_len, vocab_size, pad_index, num_layers, ffn_hidden, n_head, drop_prob, eps)
        self.projection = nn.Linear(dim, vocab_size)

    def forward(self, token_indices):
        x, attentions = self.encoder(token_indices)
        x = self.projection(x)
        return x, attentions

