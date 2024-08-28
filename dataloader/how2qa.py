import torch
from .base_dataset import BaseDataset
import pandas as pd
import json
import math


class How2QA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split="train"):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f"data/{args.dataset}/{split}.csv")
        self.features = torch.load(f"data/{args.dataset}/clipvitl14_split.pth")
        self.answer_mapping = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
        self.num_options = 4
        self.qtype_mapping = {
            "CH": 1,
            "CW": 2,
            "TN": 3,
            "TC": 4,
            "TP": 5,
            "DL": 6,
            "DC": 7,
            "DO": 8,
        }
        print(f"Num {split} data: {len(self.data)}")

    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [self.data[f"a{i}"].values[idx] for i in range(self.num_options)]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"

        a_text = "Answer: The answer is "
        text = {
            "q_text": q_text,
            "o_text": o_text,
            "a_text": a_text,
            "options": options,
        }
        return text

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            if start is not None and not math.isnan(start):
                video = self.features[video_id][int(start) : int(end) + 1].float()
            else:
                video = self.features[video_id].float()
            if not len(video):
                print(video_id, start, end)
                video = torch.zeros(1, self.features_dim)
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat(
                [video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        vid = self.data["video_id"].values[idx]
        qtype = -1
        answer = self.data["answer_id"].values[idx]
        text = self._get_text(idx)
        (
            text_id,
            label,
            video_start,
            prefix_index,
            label_mask,
            answer_index,
            question_start,
            qlen,
        ) = self._get_text_token(text, answer)
        start, end = round(self.data["start"].values[idx]), round(
            self.data["end"].values[idx]
        )
        video, video_len = self._get_video(f"{vid}", start, end)
        return {
            "vid": vid,
            "video": video,
            "video_len": video_len,
            "text": text,
            "text_id": text_id,
            "label": label,
            "video_start": video_start,
            "prefix_index": prefix_index,
            "answer_index": answer_index,
            "question_start": question_start,
            "qlen": qlen,
            "label_mask": label_mask,
            "qid": idx,
            "answer": answer,
            "qtype": qtype,
        }

    def __len__(self):
        return len(self.data)
