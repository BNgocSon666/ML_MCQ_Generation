import re
from collections import Counter


class SubwordTokenizer:
    def __init__(self, vocab_size=2000, min_freq=2):
        self.min_freq = min_freq
        self.max_vocab_size = vocab_size  # Đổi tên để tránh conflict

        # token đặc biệt
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = list(self.special_tokens)
        self.word2id = {}
        self.id2word = {}
        self.merges = []
        self.merges_set = set()  # Thêm set để tăng tốc lookup

    def get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, texts):
        # Validate input
        if not texts:
            raise ValueError("texts cannot be empty")

        # corpus char-level
        corpus = Counter()
        for line in texts:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            for word in line.split():
                corpus[' '.join(list(word)) + " </w>"] += 1

        merges = []
        vocab = set(self.special_tokens)

        while len(vocab) < self.max_vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < self.min_freq:
                break
            corpus = self.merge_vocab(best, corpus)
            merges.append(best)

        # tạo vocab từ corpus
        for word in corpus:
            for symbol in word.split():
                vocab.add(symbol)

        self.vocab = list(vocab)
        self.merges = merges
        self.merges_set = set(merges)  # Tạo set để tăng tốc lookup
        self.word2id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id2word = {i: tok for tok, i in self.word2id.items()}

    def encode_word(self, word):
        if not word:  # Handle empty word
            return []

        chars = list(word) + ["</w>"]
        i = 0
        while i < len(chars) - 1:
            pair = (chars[i], chars[i + 1])
            if pair in self.merges_set:  # Sử dụng set thay vì list - O(1) thay vì O(n)
                chars[i:i + 2] = [''.join(pair)]
            else:
                i += 1
        return chars

    def encode(self, text, add_special_tokens=True):
        if not text or not text.strip():  # Handle empty text
            if add_special_tokens:
                return [self.word2id["<bos>"], self.word2id["<eos>"]]
            return []

        ids = []
        for word in text.strip().split():
            tokens = self.encode_word(word)
            for tok in tokens:
                ids.append(self.word2id.get(tok, self.word2id["<unk>"]))

        if add_special_tokens:
            ids = [self.word2id["<bos>"]] + ids + [self.word2id["<eos>"]]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if not ids:  # Handle empty ids
            return ""

        tokens = []
        for i in ids:
            tok = self.id2word.get(i, "<unk>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            # Sửa lỗi: "</w>" có 4 ký tự
            if tok.endswith("</w>"):
                tokens.append(tok[:-4] + " ")  # Thêm space sau mỗi word
            else:
                tokens.append(tok)
        return "".join(tokens).strip()

    def get_vocab_size(self):  # Đổi tên để tránh conflict với self.max_vocab_size
        return len(self.vocab)


def pad_sequences(sequences, pad_id=0, max_len=None):
    """Pad sequences to the same length"""
    if not sequences:  # Handle empty sequences
        return []

    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [pad_id] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])  # Truncate if longer

    return padded
