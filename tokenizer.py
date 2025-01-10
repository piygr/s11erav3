import regex as re
from collections import Counter, defaultdict
import json
from tqdm import tqdm

VOCAB_SIZE = 5000 # the desired final vocabulary size

def get_stats(ids, counts=None):

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# ids: list of integer, pair: the pair of int we are merging, idx: the new int we want to replace the pair with.
def merge(ids, pair, idx):
    """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class HindiTokenizer():
    def __init__(self):
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?(?:[\u0904-\u0939\u093d-\u093d\u0950-\u0950\u0958-\u0961\u0970-\u097f\ua8f2-\ua8fe\U00011b00-\U00011b09\u1cd3-\u1cd3\u1ce9-\u1cec\u1cee-\u1cf3\u1cf5-\u1cf6\u1cfa-\u1cfa][\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\ua8e0-\ua8f1\ua8ff-\ua8ff\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced-\u1ced\u1cf4-\u1cf4\u1cf7-\u1cf9]*)+| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.special_tokens = {
            '<|endoftext|>': VOCAB_SIZE
        }

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)} # initial vocab is first 255 unicode bytes
        for (p0, p1), idx in self.merges.items(): # Get all the merges and add to vocab
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def tokenize_hindi(self, text):
        # Tokenization for Hindi, including math digits
        '''pattern = re.compile(r"""
            |[\u0900-\u097F](?![\u0964\u0965])+          # Match Hindi words (Devanagari script)
            |[\u0966-\u096F]+         # Match Hindi digits (०-९)
            |[a-zA-Z]+                # Match English words (Latin script)
            |[0-9]+                   # Match Latin digits (0-9)
            |\s+                      # Match whitespace (spaces, tabs, newlines)
            |'[^\r\n\p{L}\p{N}]*\p{L}+  # Match apostrophes followed by letters
            |\p{N}{1,3}               # Match numbers (1 to 3 digits)
            |[^\s\p{L}\p{N}]+         # Match non-letter, non-number special characters
            |\s*[\r\n]                # Match line breaks and leading spaces
            |\s+(?!\S)                # Match trailing whitespace
        """, re.VERBOSE)'''

        pattern = re.compile(self.pattern)
        return pattern.findall(text)


    def learn_bpe_vocab(self, text, num_merges=50):
        tokenized_text = self.tokenize_hindi(text)
        #print(tokenized_text)
        tokens = [list(map(int, token.encode("utf-8"))) for token in tokenized_text]
        print(len(tokens))
        input_len = 0
        for chunk_ids in tokens:
            # calculate length of tokens for compression ratio.
            # total token length is sum of all token length in each chunk.
            input_len += len(chunk_ids)

        for i in tqdm(range(num_merges), desc="Merging pairs", unit="merge"):
            stats = {}
            for chunk_ids in tokens:
                stats = get_stats(chunk_ids, stats)

            pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = [merge(chunk_ids, pair, idx) for chunk_ids in tokens]

            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        output_len = 0
        for chunk_ids in tokens:
            output_len += len(chunk_ids)

        print(f"input_len: {input_len}, output_len: {output_len} compression ratio: {input_len / output_len:.2f}X")


    def save_bpe_vocab(self, model_file):
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")


    def load_bpe_vocab(self, filepath):
        assert filepath.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(filepath, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        # get the byte for the corresponding token from vocab
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = self.tokenize_hindi(text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
