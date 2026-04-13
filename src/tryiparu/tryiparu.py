import csv
import os
import re
import string
from typing import List

import torch
from tokenizers import Tokenizer

from tryiparu.configs.config import config_g2p
from tryiparu.rules import process_text
from tryiparu.transformer import TransformerBlock

_SPLIT_RE = re.compile(r"\w+|[^\w\s]")
_PUNCTUATION = set(string.punctuation)


class G2PModel:
    def __init__(
        self,
        device: str | None = None,
        tokenizer_file: str | None = None,
        model_weights: str | None = None,
        load_dataset: bool = True,
    ) -> None:
        pkg_dir = os.path.dirname(__file__)

        if tokenizer_file is None:
            tokenizer_file = os.path.join(pkg_dir, "configs", "bpe.json")
        if model_weights is None:
            model_weights = os.path.join(pkg_dir, "weights", "model.pt")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        self.model = TransformerBlock(tokenizer=self.tokenizer, config=config_g2p)
        self.model.to(self.device)
        self.model.load_state_dict(
            torch.load(model_weights, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.max_length = config_g2p.get("MAX_LEN", 64)

        self.bos_token_id = self.tokenizer.encode("<bos>").ids[0]
        self.eos_token_id = self.tokenizer.encode("<eos>").ids[0]
        self.pad_token_id = self.tokenizer.encode("<pad>").ids[0]

        self._causal_mask = torch.tril(
            torch.ones(self.max_length, self.max_length, device=self.device)
        )

        self.data_dict: dict[str, str] = {}
        if load_dataset:
            data_path = os.path.join(pkg_dir, "data", "cleaned_dataset.csv")
            self.data_dict = self._load_dict(data_path)

    @staticmethod
    def _load_dict(path: str) -> dict[str, str]:
        result: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    result[row[0]] = row[1]
        return result

    @torch.inference_mode()
    def __call__(self, text: str) -> List[str]:
        tokens = _SPLIT_RE.findall(text.lower())
        output_tokens: list[str] = []

        for i, token in enumerate(tokens):
            if not token.isalpha():
                output_tokens.append(token)
                continue

            phoneme = self.data_dict.get(token)
            if phoneme is None:
                phoneme = self.greedy_decode(token)
                self.data_dict[token] = phoneme

            output_tokens.append(phoneme) if isinstance(phoneme, str) else output_tokens.extend(phoneme)

            if i < len(tokens) - 1 and tokens[i + 1].isalpha():
                output_tokens.append(" ")

        processed: list[str] = []
        for i, token in enumerate(output_tokens):
            if token == " ":
                if (
                    (i > 0 and output_tokens[i - 1] in _PUNCTUATION)
                    or (i < len(output_tokens) - 1 and output_tokens[i + 1] in _PUNCTUATION)
                    or (i > 0 and output_tokens[i - 1] == " ")
                ):
                    continue
            processed.append(token)

        if processed and processed[-1] == " ":
            processed.pop()

        return process_text(processed)

    @torch.inference_mode()
    def greedy_decode(self, src: str) -> List[str]:
        src_ids = self.tokenizer.encode(src).ids
        enc_len = len(src_ids) + 2
        pad_len = self.max_length - enc_len

        if pad_len < 0:
            raise ValueError(
                f"Input too long: max {self.max_length}, got {enc_len}"
            )

        enc_input = torch.zeros(1, self.max_length, dtype=torch.long, device=self.device)
        enc_input[0, 0] = self.bos_token_id
        enc_input[0, 1 : 1 + len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
        enc_input[0, 1 + len(src_ids)] = self.eos_token_id
        if pad_len > 0:
            enc_input[0, enc_len:] = self.pad_token_id

        enc_mask = (enc_input != self.pad_token_id).unsqueeze(1).unsqueeze(1).int()
        enc_output = self.model.encode(enc_input, enc_mask)

        dec_ids = torch.full(
            (1, self.max_length), self.pad_token_id, dtype=torch.long, device=self.device
        )
        dec_ids[0, 0] = self.bos_token_id
        cur_len = 1

        for _ in range(self.max_length - 1):
            tgt_mask = self._causal_mask[:cur_len, :cur_len].unsqueeze(0)
            dec_output = self.model.decode(
                enc_output, enc_mask, dec_ids[:, :cur_len], tgt_mask
            )
            logits = self.model.fc_out(dec_output[:, -1])
            next_id = logits.argmax(dim=-1).item()

            dec_ids[0, cur_len] = next_id
            cur_len += 1

            if next_id == self.eos_token_id:
                break

        decoded = self.tokenizer.decode(dec_ids[0, :cur_len].tolist())
        return self._process_decoded_output(decoded)

    @staticmethod
    def _process_decoded_output(decoded: str) -> List[str]:
        return [t for t in decoded.split("Ġ") if t]
