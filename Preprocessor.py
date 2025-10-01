import json
import torch
from torch.utils.data import Dataset, DataLoader
from Tokenizer import pad_sequences


class DataPreprocessor:
    def __init__(self, tokenizer, max_chunk_len=400, overlap_ratio=0.2):
        """
        Args:
            tokenizer: SubwordTokenizer instance
            max_chunk_len: Maximum number of words per chunk
            overlap_ratio: Ratio of overlap between chunks (0.0 - 0.5)
        """
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(max_chunk_len * overlap_ratio)

    def load_data(self, json_path):
        """Load data from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} samples from {json_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {json_path}")

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Normalize punctuation
        text = text.replace('  ', ' ')
        text = text.strip()
        
        return text

    def chunk_long_context(self, text, max_len=None):
        """
        Chia văn bản dài thành các chunks với overlap
        
        Args:
            text: Input text
            max_len: Maximum chunk length (words), defaults to self.max_chunk_len
            
        Returns:
            List of text chunks
        """
        if max_len is None:
            max_len = self.max_chunk_len
        
        text = self.clean_text(text)
        
        # Split by sentences (Vietnamese sentence endings)
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ['.', '!', '?', '。']:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        
        # If text is short enough, return as is
        total_words = sum(len(s.split()) for s in sentences)
        if total_words <= max_len:
            return [text]
        
        # Create chunks with overlap
        chunks = []
        current_chunk = []
        current_len = 0
        
        for i, sent in enumerate(sentences):
            sent_len = len(sent.split())
            
            if current_len + sent_len <= max_len:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Create overlap: take last few sentences
                overlap_chunk = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    prev_len = len(prev_sent.split())
                    if overlap_len + prev_len <= self.overlap_size:
                        overlap_chunk.insert(0, prev_sent)
                        overlap_len += prev_len
                    else:
                        break
                
                # Start new chunk with overlap + current sentence
                current_chunk = overlap_chunk + [sent]
                current_len = sum(len(s.split()) for s in current_chunk)
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [text]

    def prepare_dataset(self, json_path, train_ratio=0.8, chunk_long_texts=True):
        """
        Chuẩn bị dataset từ JSON file
        
        Args:
            json_path: Path to JSON file
            train_ratio: Ratio of training data (0.0 - 1.0)
            chunk_long_texts: Whether to chunk long contexts
            
        Returns:
            train_data, val_data (lists of dicts)
        """
        data = self.load_data(json_path)
        
        processed_data = []
        
        for item in data:
            context = item.get('context', '')
            question = item.get('question', '')
            
            if not context or not question:
                continue
            
            # Chunk long contexts if needed
            if chunk_long_texts:
                chunks = self.chunk_long_context(context)
            else:
                chunks = [context]
            
            # Create sample for each chunk
            for chunk in chunks:
                processed_data.append({
                    'context': self.clean_text(chunk),
                    'question': self.clean_text(question),
                    'options': item.get('options', []),
                    'answer': item.get('answer', '')
                })
        
        print(f"📊 Processed {len(processed_data)} samples (from {len(data)} original)")
        
        # Split train/val
        split_idx = int(len(processed_data) * train_ratio)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        
        print(f"📚 Train: {len(train_data)} samples")
        print(f"📖 Val: {len(val_data)} samples")
        
        return train_data, val_data

    def create_dataloaders(self, train_data, val_data, batch_size=8, 
                          max_src_len=512, max_tgt_len=128):
        """
        Tạo PyTorch DataLoaders
        
        Args:
            train_data: List of training samples
            val_data: List of validation samples
            batch_size: Batch size
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            
        Returns:
            train_loader, val_loader
        """
        train_dataset = QGDataset(
            train_data, 
            self.tokenizer, 
            max_src_len, 
            max_tgt_len
        )
        val_dataset = QGDataset(
            val_data, 
            self.tokenizer, 
            max_src_len, 
            max_tgt_len
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader


class QGDataset(Dataset):
    """Question Generation Dataset"""
    
    def __init__(self, data, tokenizer, max_src_len=512, max_tgt_len=128):
        """
        Args:
            data: List of dicts with 'context' and 'question'
            tokenizer: SubwordTokenizer instance
            max_src_len: Maximum source length
            max_tgt_len: Maximum target length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode context (source)
        src_ids = self.tokenizer.encode(
            item['context'], 
            add_special_tokens=True
        )
        
        # Encode question (target)
        tgt_ids = self.tokenizer.encode(
            item['question'], 
            add_special_tokens=True
        )
        
        # Truncate if too long
        src_ids = src_ids[:self.max_src_len]
        tgt_ids = tgt_ids[:self.max_tgt_len]
        
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    Pads sequences to the same length in a batch
    """
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    
    # Pad sequences
    src_ids_padded = pad_sequences(src_ids, pad_id=0)
    tgt_ids_padded = pad_sequences(tgt_ids, pad_id=0)
    
    # Convert to tensors
    src_tensor = torch.tensor(src_ids_padded, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids_padded, dtype=torch.long)
    
    # Create masks (1 for real tokens, 0 for padding)
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
    tgt_mask = create_tgt_mask(tgt_tensor)  # (B, 1, T, T)
    
    return {
        'src': src_tensor,
        'tgt': tgt_tensor,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask
    }


def create_tgt_mask(tgt):
    """
    Create target mask for decoder (causal mask + padding mask)
    
    Args:
        tgt: Target tensor (B, T)
        
    Returns:
        mask: (B, 1, T, T)
    """
    batch_size, tgt_len = tgt.size()
    
    # Padding mask: (B, 1, 1, T)
    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    
    # Causal mask: (1, 1, T, T) - prevent looking ahead
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
    
    # Combine masks
    tgt_mask = pad_mask & causal_mask.bool()
    
    return tgt_mask

