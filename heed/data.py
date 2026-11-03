"""Training data: LLaVA-style jsonl -> Qwen3-VL inputs.

One sample per line, either
    {"id": ..., "image": "rel/path.png" | ["a.png", ...] | null, "conversations": [{"from": "human"|"gpt", "value": ...}]}
or a pure-text sample
    {"id": ..., "text": ...}
`<image>` in a human turn marks where an image goes; images without a tag are put before the first user turn.
DPO samples hold the prompt in "conversations", ending on a human turn, plus "chosen" and "rejected" answers.
"""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


def to_messages(row):
    images = row.get("image") or []
    images = [images] if isinstance(images, str) else list(images)
    msgs, n_tagged = [], 0
    for turn in row["conversations"]:
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "system": "system"}[turn["from"]]
        content = []
        if role == "user":
            for j, part in enumerate(turn["value"].split("<image>")):
                if j > 0:
                    content.append({"type": "image"})
                    n_tagged += 1
                if part.strip():
                    content.append({"type": "text", "text": part.strip()})
        else:
            content.append({"type": "text", "text": turn["value"]})
        msgs.append({"role": role, "content": content})
    first_user = next(i for i, m in enumerate(msgs) if m["role"] == "user")
    msgs[first_user]["content"] = [{"type": "image"}] * (len(images) - n_tagged) + msgs[first_user]["content"]
    return msgs, images


class DistillDataset(Dataset):
    """App. C.3 caps: text-only samples at 4,096 tokens, vision-text samples at 8,192 and 2,500 visual tokens/image."""

    def __init__(self, path, processor, image_root=None, max_len=8192, max_text_len=4096, max_visual_tokens=2500):
        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.processor, self.tok = processor, processor.tokenizer
        self.image_root = image_root or os.path.dirname(os.path.abspath(path))
        self.max_len, self.max_text_len = max_len, max_text_len
        ip = processor.image_processor
        px = ip.patch_size * ip.merge_size
        ip.size = {"shortest_edge": ip.size["shortest_edge"], "longest_edge": max_visual_tokens * px * px}
        self.image_token_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.im_start = self.tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant = self.tok.convert_tokens_to_ids("assistant")

    def __len__(self):
        return len(self.rows)

    def assistant_mask(self, ids):
        """True on assistant reply tokens (including <|im_end|>), the positions that KD and CE supervise."""
        mask = torch.zeros(len(ids), dtype=torch.bool)
        i, ids_l = 0, ids.tolist()
        while i < len(ids_l):
            if ids_l[i] == self.im_start and ids_l[i + 1:i + 2] == [self.assistant]:
                j = i + 3  # past the newline, which merges with any newline the reply starts with
                while j < len(ids_l) and ids_l[j] != self.im_end:
                    j += 1
                mask[i + 3:j + 1] = True
                i = j
            i += 1
        return mask

    def __getitem__(self, idx):
        return self.encode(self.rows[idx])

    def encode(self, row):
        if "text" in row:
            ids = self.tok(row["text"], truncation=True, max_length=self.max_text_len,
                           return_tensors="pt")["input_ids"][0]
            return {"id": str(row["id"]), "input_ids": ids, "loss_mask": torch.ones_like(ids, dtype=torch.bool),
                    "mm_token_type_ids": torch.zeros_like(ids), "pixel_values": None, "image_grid_thw": None}

        msgs, image_paths = to_messages(row)
        text = self.processor.apply_chat_template(msgs, tokenize=False)
        images = [Image.open(os.path.join(self.image_root, p)).convert("RGB") for p in image_paths] or None
        enc = self.processor(text=[text], images=images, return_tensors="pt")
        ids, mm_types = enc["input_ids"][0], enc["mm_token_type_ids"][0]
        loss_mask = self.assistant_mask(ids)
        if len(ids) > self.max_len:
            if (ids[self.max_len:] == self.image_token_id).any():
                return None  # truncation would cut into an image
            ids, loss_mask, mm_types = ids[:self.max_len], loss_mask[:self.max_len], mm_types[:self.max_len]
        return {"id": str(row["id"]), "input_ids": ids, "loss_mask": loss_mask, "mm_token_type_ids": mm_types,
                "pixel_values": enc.get("pixel_values"), "image_grid_thw": enc.get("image_grid_thw")}


class PairDataset(DistillDataset):
    def __getitem__(self, idx):
        row = self.rows[idx]
        pair = [self.encode({**row, "conversations": row["conversations"] + [{"from": "gpt", "value": row[k]}]})
                for k in ("chosen", "rejected")]
        return None if None in pair else pair


def collate_pairs(batch, pad_id):
    """The chosen answers, then the rejected ones in the same order."""
    pairs = [p for p in batch if p is not None]
    return collate([p[0] for p in pairs] + [p[1] for p in pairs], pad_id)


def collate(batch, pad_id):
    """Right padding: the Mamba-2 scan is causal, so trailing pad tokens never reach a valid position."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    L = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), L), dtype=torch.long)
    loss_mask = torch.zeros((len(batch), L), dtype=torch.bool)
    mm_token_type_ids = torch.zeros((len(batch), L), dtype=torch.long)  # 0 text, 1 image, 2 video (M-RoPE)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = 1
        loss_mask[i, :n] = b["loss_mask"]
        mm_token_type_ids[i, :n] = b["mm_token_type_ids"]
    with_images = [b for b in batch if b["pixel_values"] is not None]
    return {
        "ids": [b["id"] for b in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": torch.cat([b["pixel_values"] for b in with_images]) if with_images else None,
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in with_images]) if with_images else None,
        "num_images": [0 if b["image_grid_thw"] is None else len(b["image_grid_thw"]) for b in batch],
    }
