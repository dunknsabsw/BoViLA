import torch
from torch.utils.data import Dataset
import copy


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.max_feats = args.max_feats
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split

    def _get_padding_id(self, text_id):
        padding_text_id = (
            torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        )
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, : len(tid)] = tid
            else:
                padding_text_id[i] = tid[: self.max_seq_len]
                print("max sequence length overflow")
        return padding_text_id

    def _get_text_token(self, text, answer):
        (
            vqa_id,
            vqa_prefix_index,
            vqa_video_start,
            vqa_question_start,
            vqa_qlen,
            vqa_answer_index,
        ) = self.tokenizer.encode_vqa(
            text=text,
            max_feats=self.max_feats,
            split=self.split,
            answer_mapping=self.answer_mapping,
            answer=answer,
            args=self.args,
        )
        vaq_id, vaq_prefix_index, vaq_video_start, vaq_answer_index = (
            self.tokenizer.encode_vaq(
                text=text,
                max_feats=self.max_feats,
                split=self.split,
                answer_mapping=self.answer_mapping,
                answer=answer,
                args=self.args,
            )
        )

        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vaq_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vaq_id]

        vqa_padding_text_id = self._get_padding_id(vqa_id)
        vaq_padding_text_id = self._get_padding_id(vaq_id)

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float()

        vaq_label = copy.deepcopy(vaq_padding_text_id)
        vaq_label[:, :vaq_prefix_index] = -1
        vaq_label_mask = vaq_label.ge(0)
        vaq_label[~vaq_label_mask] = 0
        vaq_label_mask = vaq_label_mask.float()

        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
        vaq_text_mask = vaq_padding_text_id.ge(0)
        vaq_padding_text_id[~vaq_text_mask] = 0

        text_id = {
            "vqa": vqa_padding_text_id,
            "vaq": vaq_padding_text_id,
        }
        label = {
            "vqa": vqa_label,
            "vaq": vaq_label,
        }
        video_start = {
            "vqa": vqa_video_start,
            "vaq": vaq_video_start,
        }
        answer_index = {
            "vqa": vqa_answer_index,
            "vaq": vaq_answer_index,
        }
        prefix_index = {
            "vqa": vqa_prefix_index,
            "vaq": vaq_prefix_index,
        }
        label_mask = {
            "vqa": vqa_label_mask,
            "vaq": vaq_label_mask,
        }
        question_start = {
            "vqa": vqa_question_start,
        }
        qlen = {
            "vqa": vqa_qlen,
        }

        return (
            text_id,
            label,
            video_start,
            prefix_index,
            label_mask,
            answer_index,
            question_start,
            qlen,
        )
