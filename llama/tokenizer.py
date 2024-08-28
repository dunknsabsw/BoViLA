# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from .llama3_tokenizer import Tokenizer as Tokenizer3
from sentencepiece import SentencePieceProcessor

from logging import getLogger
from typing import List
import os
import torch
import random

logger = getLogger()


class Tokenizer_3:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.tk_model = Tokenizer3(model_path=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.tk_model.n_words
        self.bos_id: int = self.tk_model.bos_id
        self.eos_id: int = self.tk_model.eos_id
        self.pad_id: int = self.tk_model.pad_id
        self.unk_id: int = self.tk_model.pad_id
        
        self.v_token_id = 10955
        self.q_token_id = 14924
        self.a_token_id = 16533
        self.nl_id = 198
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tk_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_vqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None, args=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.tk_model.encode(s1)
        video_start = len(t1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.tk_model.encode(s2) + [self.eos_id]
            t = [t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.tk_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.a_token_id) + 5

        answer_index = t[0].index(self.tk_model.encode(answer_mapping[answer])[1])
        question_start = t[0].index(14924) + 2
        qlen = len(self.tk_model.encode(q_text)) - 2

        return t, prefix_index, video_start, question_start, qlen, answer_index

    def encode_dvqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the dialogue, video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
     
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.tk_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.tk_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.tk_model.encode(s2) + [self.eos_id]
            t = [t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.tk_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + d1 + t2)

        question_start = t[0].index(14924) + 2 # TODO: What if more than one "Question"?
        prefix_index = len(t[0]) - 4
        answer_index = t[0].index(self.tk_model.encode(answer_mapping[answer])[1])
        qlen = len(self.tk_model.encode(q_text)) - 2
        return t, prefix_index, video_start, question_start, qlen, answer_index, prefix_i, prefix_main
    
    def encode_vaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None, args=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.tk_model.encode(s1)
        video_start = len(t1)
        
        s2 = o_text + a_text
        
        s2 = s2 + answer_mapping[answer] + "\n" + q_text
        t2 = self.tk_model.encode(s2) + [self.eos_id]
        t = [t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + t2]
        prefix_index = t[0].index(self.q_token_id) + 2

        answer_index = t[0].index(self.tk_model.encode(answer_mapping[answer])[1])

        return t, prefix_index, video_start, answer_index
    
    def encode_dvaq(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the question based on the dialogue, video and answer.\n"
        q_text = text['q_text'].strip()
        o_text = text['o_text']
        a_text = text['a_text']
        d_text = text['d_text']
        
        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.tk_model.encode(s1)
        video_start = len(t1)
        
        prefix_i = video_start + max_feats + 1
        d1 = self.tk_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = o_text + a_text
        
        if split == 'train':
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.tk_model.encode(s2) + [self.eos_id]
            t = [t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.tk_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + d1 + t2)
        
        prefix_index = t[0].index(self.q_token_id) + 2
        answer_index = t[0].index(self.tk_model.encode(answer_mapping[answer])[1])
        return t, prefix_index, video_start, answer_index, prefix_i, prefix_main

    def decode(self, t: List[int]) -> str:
        return self.tk_model.decode(t)
    

class Tokenizer_1:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        self.v_token_id = 15167
        self.q_token_id = 16492
        self.a_token_id = 22550
        self.nl_id = 13
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_vqa(
        self,
        text=None,
        max_feats=10,
        split="train",
        answer_mapping=None,
        answer=None,
        args=None,
    ) -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text["q_text"]
        o_text = text["o_text"]
        a_text = text["a_text"]

        s1 = i_text + "Video:"
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        s2 = q_text + o_text + a_text  # ot
        # s2 = '\n' + q_text + o_text + a_text # nt

        if split == "train":
            s2 = s2 + answer_mapping[answer]
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]  # ot
            # t = [t1 + [-2 for _ in range(max_feats)] + t2] # nt
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2)  # ot
                # t.append(t1 + [-2 for _ in range(max_feats)] + t2) # nt
            prefix_index = t[answer].index(self.a_token_id) + 5

        answer_index = t[0].index(self.sp_model.encode(answer_mapping[answer])[1])

        question_start = t[0].index(894) + 2  # ot
        # question_start = t[0].index(self.q_token_id) + 2 # nt

        qlen = len(self.sp_model.encode(q_text)) - 2

        return t, prefix_index, video_start, question_start, qlen, answer_index

    def encode_dvqa(
        self, text=None, max_feats=10, split="train", answer_mapping=None, answer=None
    ) -> List[int]:
        i_text = "Instruction: Predict the answer based on the dialogue, video and question.\n"
        q_text = text["q_text"]
        o_text = text["o_text"]
        a_text = text["a_text"]
        d_text = text["d_text"]

        s1 = i_text + "Video:"
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = q_text + o_text + a_text

        if split == "train":
            s2 = s2 + answer_mapping[answer]
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)

        question_start = t[0].index(894) + 2  # TODO: What if more than one "Question"?
        prefix_index = len(t[0]) - 4
        answer_index = t[0].index(self.sp_model.encode(answer_mapping[answer])[1])
        qlen = len(self.sp_model.encode(q_text)) - 2
        return (
            t,
            prefix_index,
            video_start,
            question_start,
            qlen,
            answer_index,
            prefix_i,
            prefix_main,
        )

    def encode_odvqa(
        self, text=None, max_feats=10, split="train", answer=None
    ) -> List[int]:
        i_text = "Instruction: Predict the answer based on the dialogue, video and question.\n"
        q_text = text["q_text"]
        o_text = text["o_text"]
        a_text = text["a_text"]
        d_text = text["d_text"]

        s1 = i_text + "Video:"
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = q_text + o_text + a_text

        if "train" in split:
            s2 = s2 + answer
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]

        question_start = t[0].index(894) + 2  # TODO: What if more than one "Question"?
        prefix_index = len(t[0]) - 4
        answer_index = t[0].index(self.sp_model.encode(answer)[0])
        qlen = len(self.sp_model.encode(q_text)) - 2
        return (
            t,
            prefix_index,
            video_start,
            question_start,
            qlen,
            answer_index,
            prefix_i,
            prefix_main,
        )

    def encode_vaq(
        self,
        text=None,
        max_feats=10,
        split="train",
        answer_mapping=None,
        answer=None,
        args=None,
    ) -> List[int]:
        i_text = "Instruction: Predict the question based on the video and answer.\n"
        q_text = text["q_text"].strip()
        o_text = text["o_text"]
        a_text = text["a_text"]

        s1 = i_text + "Video:"
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        s2 = o_text + a_text

        s2 = s2 + answer_mapping[answer] + "\n" + q_text
        t2 = self.sp_model.encode(s2) + [self.eos_id]
        t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + t2]
        prefix_index = t[0].index(self.q_token_id) + 2

        answer_index = t[0].index(self.sp_model.encode(answer_mapping[answer])[1])

        return t, prefix_index, video_start, answer_index

    def encode_dvaq(
        self, text=None, max_feats=10, split="train", answer_mapping=None, answer=None
    ) -> List[int]:
        i_text = "Instruction: Predict the question based on the dialogue, video and answer.\n"
        q_text = text["q_text"].strip()
        o_text = text["o_text"]
        a_text = text["a_text"]
        d_text = text["d_text"]

        s1 = i_text + "Video:"
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = o_text + a_text

        if split == "train":
            s2 = s2 + answer_mapping[answer] + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v + "\n" + q_text) + [self.eos_id]
                t.append(t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2)

        prefix_index = t[0].index(self.q_token_id) + 2
        answer_index = t[0].index(self.sp_model.encode(answer_mapping[answer])[1])
        return t, prefix_index, video_start, answer_index, prefix_i, prefix_main

    def encode_odvaq(
        self, text=None, max_feats=10, split="train", answer=None
    ) -> List[int]:
        i_text = "Instruction: Predict the question based on the dialogue, video and answer.\n"
        q_text = text["q_text"].strip()
        o_text = text["o_text"]
        a_text = text["a_text"]
        d_text = text["d_text"]

        s1 = i_text + "Video:"
        t1 = [self.bos_id] + self.sp_model.encode(s1)
        video_start = len(t1)

        prefix_i = video_start + max_feats + 1
        d1 = self.sp_model.encode(d_text)
        prefix_main = prefix_i + len(d1)

        s2 = o_text + a_text

        if "train" in split:
            s2 = s2 + answer + "\n" + q_text
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [-2 for _ in range(max_feats)] + [self.nl_id] + d1 + t2]

        prefix_index = t[0].index(self.q_token_id) + 2
        answer_index = t[0].index(self.sp_model.encode(answer)[0])
        return t, prefix_index, video_start, answer_index, prefix_i, prefix_main

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
