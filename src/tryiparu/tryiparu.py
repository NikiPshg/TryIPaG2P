import os
import re
import string
from typing import List
import pandas as pd

import torch
from tokenizers import Tokenizer

from tryiparu.configs.config import config_g2p
from tryiparu.transformer import TransformerBlock
from tryiparu.rules import process_text

ABSOLUTE_PATH = os.path.abspath(os.path.dirname(__file__))


class G2PModel:
    def __init__(
        self,
        tokenizer_file: str = os.path.join(ABSOLUTE_PATH, "configs", "bpe_512_kaikii_clean"),
        model_weights: str = os.path.join(ABSOLUTE_PATH, "weight", "model.pt"),
        max_length: int = 64,
        load_dataset: bool = True
    ) -> None:
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        self.model = TransformerBlock(tokenizer=self.tokenizer, config=config_g2p)
        self.model.load_state_dict(torch.load(model_weights, map_location="cpu"))
        self.model.eval()
        self.max_length = max_length

        self.bos_token_id = self.tokenizer.encode("<bos>").ids[0]
        self.eos_token_id = self.tokenizer.encode("<eos>").ids[0]
        self.pad_token_id = self.tokenizer.encode("<pad>").ids[0]

        self.data_dict = {}
        if load_dataset:
            data_path = os.path.join(ABSOLUTE_PATH, 'data/cleaned_dataset.csv')
            df = pd.read_csv(data_path)
            self.data_dict = df.set_index('words')['phonemes'].to_dict()

    @torch.inference_mode()
    def __call__(self, text: str) -> List[str]:
        tokens = self._split_text(text.lower())
        output_tokens = []

        for i, token in enumerate(tokens):
            if not token.isalpha():
                output_tokens.append(token)
            else:
                phoneme_tokens = self.data_dict.get(
                    token,
                    self.greedy_decode(
                        src=token,
                        max_length=self.max_length
                    )
                )
                phoneme_tokens = [phoneme_tokens] if not isinstance(phoneme_tokens, list) else phoneme_tokens
                output_tokens.extend(phoneme_tokens)

                if i < len(tokens) - 1 and tokens[i + 1].isalpha():
                    output_tokens.append(" ")

        processed_tokens = []
        for i, token in enumerate(output_tokens):
            if token == " ":
                if (
                    (i > 0 and output_tokens[i - 1] in string.punctuation)
                    or (i < len(output_tokens) - 1 and output_tokens[i + 1] in string.punctuation)
                    or (i > 0 and output_tokens[i - 1] == " ")
                ):
                    continue
            processed_tokens.append(token)

        # Убираем пробел в конце, если он есть
        if processed_tokens and processed_tokens[-1] == " ":
            processed_tokens.pop()

        return process_text(processed_tokens)

    @torch.inference_mode()
    def greedy_decode(self, src: str, max_length: int) -> List[str]:
        src_tokens = self.tokenizer.encode(src).ids
        encoder_sequence_length = len(src_tokens) + 2
        padding_length = self.max_length - encoder_sequence_length

        encoder_input = torch.cat([
            torch.tensor([self.bos_token_id]),
            torch.tensor(src_tokens),
            torch.tensor([self.eos_token_id]),
            torch.tensor([self.pad_token_id] * padding_length),
        ], dim=0)

        encoder_input = encoder_input.unsqueeze(0)
        encoder_mask = (encoder_input != self.pad_token_id).unsqueeze(1).unsqueeze(1).int()

        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_input = torch.tensor([[self.bos_token_id]], dtype=encoder_input.dtype)

        for _ in range(max_length - 1):
            tgt_mask = torch.tril(
                torch.ones((decoder_input.size(1), decoder_input.size(1)), dtype=encoder_input.dtype)
            ).unsqueeze(0)
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, tgt_mask)
            logits = self.model.fc_out(decoder_output[:, -1])
            next_token = torch.argmax(logits, dim=1).item()

            decoder_input = torch.cat([
                decoder_input,
                torch.tensor([[next_token]], dtype=encoder_input.dtype)
            ], dim=1)

            if next_token == self.eos_token_id:
                break

        decoded_str = self.tokenizer.decode(decoder_input[0].tolist())
        phoneme_tokens = self._process_decoded_output(decoded_str)

        return phoneme_tokens
    

    @staticmethod
    def _split_text(text: str) -> List[str]:
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    @staticmethod
    def _process_decoded_output(decoded: str) -> List[str]:
        return [token for token in decoded.split("Ġ") if token]


if __name__ == "__main__":
    pass