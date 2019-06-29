import numpy as np
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class Tokenizer:
    def __init__(self, max_seq_len: int, bert_vocab_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_path)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text: str, reverse: bool = False,
                         pad: str = 'post',
                         truncate: str = 'post') -> np.ndarray:
        tokens = self.tokenizer.tokenize(text)
        idx_seq = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(idx_seq) == 0:
            idx_seq = [0]
        if reverse:
            idx_seq = idx_seq[::-1]
        aligned = self.align(idx_seq, self.max_seq_len, truncate, pad)
        return aligned

    @staticmethod
    def align(seq: np.ndarray, max_len: int, truncate: str = 'pre',
              pad: str = 'post') -> np.ndarray:
        # Truncate data
        if truncate == 'post':
            truncated = seq[:max_len]
        else:
            truncated = seq[-max_len:]
        # Pad data
        padded = (np.ones(max_len) * 0).astype('int64')
        if pad == 'post':
            padded[:len(truncated)] = truncated
        else:
            padded[-len(truncated):] = truncated
        return padded


class SentimentDataset(Dataset):
    def __init__(self, fname: str, tokenizer: Tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in
                                        lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(
                text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1

            text_bert_indices = tokenizer.text_to_sequence(
                ' '.join(['[CLS]', text_left, aspect, text_right,
                          '[SEP]', aspect, '[SEP]']))
            bert_segments_ids = np.asarray(
                [0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (
                        aspect_len + 1))
            bert_segments_ids = tokenizer.align(bert_segments_ids,
                                                tokenizer.max_seq_len)
            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'polarity': polarity,
            }
            all_data.append(data)
        self.__data = all_data

    def __getitem__(self, index):
        return self.__data[index]

    def __len__(self):
        return len(self.__data)
