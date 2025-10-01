import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import json
import os
import time
from datetime import datetime
from Tokenizer import SubwordTokenizer
from Preprocessor import DataPreprocessor
from Transformer import Transformer


class Trainer:
    def __init__(self, config):
        """
        Initialize Trainer

        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Disable FP16 if no CUDA
        if not torch.cuda.is_available() and config.get('use_fp16', False):
            print("⚠️  CUDA not available, disabling FP16")
            config['use_fp16'] = False

        print("=" * 70)
        print("🚀 INITIALIZING TRAINER")
        print("=" * 70)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  Running on CPU (training will be slower)")

        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = GradScaler('cuda') if config['use_fp16'] and torch.cuda.is_available() else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def prepare_tokenizer(self, data_path):
        """Train or load tokenizer"""
        print("\n" + "=" * 70)
        print("📝 PREPARING TOKENIZER")
        print("=" * 70)

        tokenizer_path = os.path.join(self.config['checkpoint_dir'], 'tokenizer.json')

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create corpus
        texts = []
        for item in data:
            texts.append(item['context'])
            texts.append(item['question'])

        # Train tokenizer
        self.tokenizer = SubwordTokenizer(
            vocab_size=self.config['vocab_size'],
            min_freq=self.config['min_freq']
        )
        self.tokenizer.train(texts)

        print(f"✅ Tokenizer trained")
        print(f"   Vocabulary size: {self.tokenizer.get_vocab_size()}")

        # Save tokenizer
        self.save_tokenizer(tokenizer_path)

        return self.tokenizer

    def save_tokenizer(self, path):
        """Save tokenizer to file"""
        tokenizer_data = {
            'vocab': self.tokenizer.vocab,
            'word2id': self.tokenizer.word2id,
            'id2word': self.tokenizer.id2word,
            'merges': self.tokenizer.merges,
            'special_tokens': self.tokenizer.special_tokens,
            'max_vocab_size': self.tokenizer.max_vocab_size,
            'min_freq': self.tokenizer.min_freq
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Tokenizer saved to {path}")

    def load_tokenizer(self, path):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        self.tokenizer = SubwordTokenizer(
            vocab_size=tokenizer_data['max_vocab_size'],
            min_freq=tokenizer_data['min_freq']
        )
        self.tokenizer.vocab = tokenizer_data['vocab']
        self.tokenizer.word2id = tokenizer_data['word2id']
        self.tokenizer.id2word = {int(k): v for k, v in tokenizer_data['id2word'].items()}
        self.tokenizer.merges = [tuple(m) for m in tokenizer_data['merges']]
        self.tokenizer.merges_set = set(self.tokenizer.merges)
        self.tokenizer.special_tokens = tokenizer_data['special_tokens']

        print(f"✅ Tokenizer loaded from {path}")
        return self.tokenizer

    def build_model(self):
        """Build Transformer model"""
        print("\n" + "=" * 70)
        print("🏗️  BUILDING MODEL")
        print("=" * 70)

        self.model = Transformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            hidden_dim=self.config['hidden_dim'],
            dropout=self.config['dropout'],
            max_len=self.config['max_len']
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✅ Model built")
        print(f"   Total parameters: {num_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{num_params * 4 / 1e6:.2f} MB")

        return self.model

    def setup_training(self):
        """Setup optimizer and loss function"""
        print("\n" + "=" * 70)
        print("⚙️  SETUP TRAINING")
        print("=" * 70)

        # Loss function (ignore padding token)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.word2id['<pad>'],
            label_smoothing=self.config.get('label_smoothing', 0.0)
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        print(f"✅ Training setup complete")
        print(f"   Optimizer: AdamW")
        print(f"   Learning rate: {self.config['learning_rate']}")
        print(f"   Loss function: CrossEntropyLoss")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_mask = batch['src_mask'].to(self.device)
            tgt_mask = batch['tgt_mask'].to(self.device)

            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :, :-1, :-1]

            # Forward pass
            if self.config['use_fp16']:
                with autocast(device_type='cuda'):
                    output = self.model(src, tgt_input, src_mask, tgt_mask)
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)),
                        tgt_output.reshape(-1)
                    )

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

                self.optimizer.step()

            total_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"   Batch [{batch_idx + 1}/{num_batches}] Loss: {avg_loss:.4f}")

        return total_loss / num_batches

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                tgt_mask = batch['tgt_mask'].to(self.device)

                # Prepare target
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_mask = tgt_mask[:, :, :-1, :-1]

                # Forward pass
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )

                total_loss += loss.item()

        return total_loss / num_batches

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        print(f"   Resuming from epoch {self.current_epoch}")

    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("🎓 STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['num_epochs']):
            epoch_start = time.time()

            print(f"\n📅 Epoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 70)

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Calculate time
            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print("-" * 70)
            print(f"📊 Epoch {epoch + 1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Time: {epoch_time:.2f}s")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"   🎉 New best validation loss!")

            if (epoch + 1) % self.config.get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)

            self.current_epoch = epoch + 1

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        print(f"Final val loss: {self.val_losses[-1]:.4f}")

        # Save final model
        self.save_checkpoint(self.current_epoch, self.val_losses[-1], False)


def main():
    """Main function to run training"""

    # Configuration - Optimized for GTX 1650 (4GB VRAM)
    config = {
        # Data
        'data_path': 'Datasets.json',
        'train_ratio': 0.8,

        # Tokenizer
        'vocab_size': 2000,
        'min_freq': 2,

        # Model architecture - Small config for GTX 1650
        'd_model': 128,           # Reduced from 256
        'num_heads': 4,           # Reduced from 8
        'num_layers': 3,          # Reduced from 4
        'hidden_dim': 256,        # Reduced from 512
        'dropout': 0.1,
        'max_len': 512,           # Max sequence length

        # Training
        'num_epochs': 50,
        'batch_size': 4,          # Small batch for 4GB VRAM
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'label_smoothing': 0.1,

        # Data loading
        'max_src_len': 256,
        'max_tgt_len': 64,

        # Optimization
        'use_fp16': True,         # Use mixed precision for GTX 1650

        # Logging & Checkpointing
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'save_interval': 5,
        'log_interval': 5,
    }

    print("=" * 70)
    print("🚀 VIETNAMESE QUESTION GENERATION TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Initialize trainer
    trainer = Trainer(config)

    # Prepare tokenizer
    trainer.prepare_tokenizer(config['data_path'])

    # Build model
    trainer.build_model()

    # Setup training
    trainer.setup_training()

    # Prepare data
    print("\n" + "=" * 70)
    print("📦 PREPARING DATA")
    print("=" * 70)

    preprocessor = DataPreprocessor(
        trainer.tokenizer,
        max_chunk_len=400,
        overlap_ratio=0.2
    )

    train_data, val_data = preprocessor.prepare_dataset(
        config['data_path'],
        train_ratio=config['train_ratio'],
        chunk_long_texts=False  # Dataset is already short
    )

    train_loader, val_loader = preprocessor.create_dataloaders(
        train_data,
        val_data,
        batch_size=config['batch_size'],
        max_src_len=config['max_src_len'],
        max_tgt_len=config['max_tgt_len']
    )

    # Start training
    trainer.train(train_loader, val_loader)

    print("\n" + "=" * 70)
    print("🎉 ALL DONE!")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Next steps:")
    print("1. Check 'checkpoints/best_model.pt' for the best model")
    print("2. Use test_generation.py to test the trained model")
    print("3. Adjust hyperparameters if needed and retrain")


if __name__ == "__main__":
    main()
