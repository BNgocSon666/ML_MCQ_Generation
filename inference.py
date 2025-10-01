"""
Inference script - Load trained model và tạo câu hỏi từ context mới
Không cần train lại, chỉ cần load checkpoint
"""
import torch
import json
import os
from Tokenizer import SubwordTokenizer
from Transformer import Transformer


class QuestionGenerator:
    def __init__(self, checkpoint_path='checkpoints/best_model.pt', 
                 tokenizer_path='checkpoints/tokenizer.json'):
        """
        Initialize Question Generator
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            tokenizer_path: Path to saved tokenizer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 70)
        print("🤖 LOADING QUESTION GENERATOR")
        print("=" * 70)
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        
        # Load model
        self.model, self.config = self.load_model(checkpoint_path)
        
        print("✅ Question Generator ready!")
        print("=" * 70)
    
    def load_tokenizer(self, path):
        """Load tokenizer from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer not found: {path}")
        
        print(f"\n📝 Loading tokenizer from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = SubwordTokenizer(
            vocab_size=tokenizer_data['max_vocab_size'],
            min_freq=tokenizer_data['min_freq']
        )
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.word2id = tokenizer_data['word2id']
        tokenizer.id2word = {int(k): v for k, v in tokenizer_data['id2word'].items()}
        tokenizer.merges = [tuple(m) for m in tokenizer_data['merges']]
        tokenizer.merges_set = set(tokenizer.merges)
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
        
        return tokenizer
    
    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"\n🏗️  Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Build model
        model = Transformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            hidden_dim=config['hidden_dim'],
            dropout=0.0,  # No dropout for inference
            max_len=config['max_len']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded")
        print(f"   Parameters: {num_params:,}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        
        return model, config
    
    def generate(self, context, max_len=50, method='greedy', temperature=1.0, top_k=50):
        """
        Generate question from context
        
        Args:
            context: Input context string
            max_len: Maximum generation length
            method: 'greedy', 'sampling', or 'top_k'
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            
        Returns:
            Generated question string
        """
        self.model.eval()
        
        # Encode context
        src_ids = self.tokenizer.encode(context, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        
        # Encode
        with torch.no_grad():
            enc_output = self.model.encode(src_tensor, src_mask=None)
        
        # Start with <bos> token
        tgt_ids = [self.tokenizer.word2id["<bos>"]]
        
        # Generate
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Decode
                dec_output = self.model.decode(tgt_tensor, enc_output)
                
                # Get logits for next token
                logits = self.model.fc_out(dec_output)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Select next token based on method
                if method == 'greedy':
                    next_token = next_token_logits.argmax().item()
                elif method == 'sampling':
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                elif method == 'top_k':
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Stop if <eos>
                if next_token == self.tokenizer.word2id["<eos>"]:
                    break
                
                tgt_ids.append(next_token)
        
        # Decode to text
        question = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
        
        return question
    
    def generate_batch(self, contexts, max_len=50, method='greedy'):
        """Generate questions for multiple contexts"""
        questions = []
        for context in contexts:
            question = self.generate(context, max_len, method)
            questions.append(question)
        return questions


def main():
    """Demo usage"""
    print("\n" + "=" * 70)
    print("🎯 VIETNAMESE QUESTION GENERATOR - INFERENCE")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists('checkpoints/best_model.pt'):
        print("\n❌ ERROR: No trained model found!")
        print("Please train the model first by running:")
        print("   python Train.py")
        return
    
    # Initialize generator
    generator = QuestionGenerator(
        checkpoint_path='checkpoints/best_model.pt',
        tokenizer_path='checkpoints/tokenizer.json'
    )
    
    # Test contexts
    test_contexts = [
        "Ở người, quá trình tiêu hoá hoá học protein bắt đầu từ dạ dày.",
        "Photosynthesis là quá trình mà thực vật sử dụng ánh sáng mặt trời để chuyển đổi carbon dioxide và nước thành glucose và oxygen.",
        "Chiến tranh Việt Nam kết thúc vào ngày 30 tháng 4 năm 1975 khi quân giải phóng tiến vào Dinh Độc Lập.",
        "Liên hợp quốc được thành lập vào năm 1945 với mục tiêu duy trì hòa bình và an ninh quốc tế.",
    ]
    
    print("\n" + "=" * 70)
    print("📝 GENERATING QUESTIONS")
    print("=" * 70)
    
    for i, context in enumerate(test_contexts):
        print(f"\n🔹 Context {i+1}:")
        print(f"   {context}")
        
        # Greedy decoding
        question_greedy = generator.generate(context, max_len=50, method='greedy')
        print(f"\n   ✅ Question (Greedy):")
        print(f"   {question_greedy}")
        
        # Sampling (more diverse)
        question_sampling = generator.generate(context, max_len=50, method='sampling', temperature=0.8)
        print(f"\n   ✅ Question (Sampling):")
        print(f"   {question_sampling}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 70)
    print("\n💡 Usage in your code:")
    print("""
from inference import QuestionGenerator

# Load model once
generator = QuestionGenerator()

# Generate questions
context = "Your context here..."
question = generator.generate(context)
print(question)

# Generate with different methods
question_greedy = generator.generate(context, method='greedy')
question_diverse = generator.generate(context, method='sampling', temperature=0.8)
question_topk = generator.generate(context, method='top_k', top_k=50)
    """)


if __name__ == "__main__":
    main()

