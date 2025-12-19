# Deep Past Challenge: Akkadian-to-English Translation
## Complete Implementation Guide

---

## Table of Contents
1. [Challenge Overview](#1-challenge-overview)
2. [Technical Analysis](#2-technical-analysis)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [Model Architecture Options](#4-model-architecture-options)
5. [Training Strategy](#5-training-strategy)
6. [Inference & Submission](#6-inference--submission)
7. [Complete Code Implementation](#7-complete-code-implementation)
8. [Tips for Winning](#8-tips-for-winning)

---

## 1. Challenge Overview

### What You're Building
A neural machine translation (NMT) system that translates **transliterated Akkadian** (Old Assyrian dialect) into **English**.

### Key Constraints
| Constraint | Value |
|------------|-------|
| Runtime | ≤ 9 hours (CPU or GPU) |
| Internet | Disabled during inference |
| Output | `submission.csv` with `id,translation` |
| Evaluation | Geometric Mean of BLEU × chrF++ |

### Why This Is Hard
- **Low-resource language**: Only ~8,000 training examples
- **Morphologically rich**: One Akkadian word = multiple English words
- **Complex formatting**: Superscripts, subscripts, brackets, gaps
- **Domain-specific**: Ancient merchant/commercial vocabulary
- **OCR noise**: Translations were digitized with errors

---

## 2. Technical Analysis

### Understanding the Evaluation Metric

```python
import sacrebleu

def compute_score(predictions, references):
    """
    Geometric mean of BLEU and chrF++
    Both are corpus-level (micro-averaged)
    """
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)  # chrF++
    
    geometric_mean = (bleu.score * chrf.score) ** 0.5
    return geometric_mean
```

### Key Insight
- **BLEU** rewards exact n-gram matches (precision-focused)
- **chrF++** rewards character-level matches (more forgiving for morphology)
- The geometric mean means you need to be good at BOTH

### What This Means for Your Model
1. Get proper nouns right (exact matches help BLEU)
2. Get morphological endings close (chrF++ is forgiving)
3. Don't hallucinate extra content (hurts precision)

---

## 3. Data Preprocessing Pipeline

### 3.1 Text Cleaning Functions

```python
import re
from typing import List, Tuple

class AkkadianPreprocessor:
    """
    Preprocessor for Akkadian transliterations
    Based on competition guidelines
    """
    
    def __init__(self):
        # Determinatives to normalize (keep in curly braces)
        self.determinatives = ['ki', 'd', 'f', 'm', 'urudu', 'kug']
        
        # Superscript/subscript number patterns
        self.subscript_pattern = re.compile(r'([a-zA-Z]+)([₀-₉]+)')
        self.superscript_pattern = re.compile(r'([a-zA-Z]+)([⁰-⁹]+)')
        
    def clean_transliteration(self, text: str) -> str:
        """Clean Akkadian transliteration text"""
        
        # Remove modern scribal notations
        text = re.sub(r'[!?/]', '', text)  # Certainty markers, line dividers
        text = re.sub(r'[:\.]', ' ', text)  # Word dividers -> spaces
        
        # Handle brackets - keep content, remove brackets
        text = re.sub(r'<([^>]+)>', r'\1', text)  # Scribal insertions
        text = re.sub(r'<<([^>]+)>>', '', text)   # Erroneous signs (remove)
        text = re.sub(r'[˹˺]', '', text)          # Partial breaks
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)  # Square brackets
        
        # Handle gaps and breaks
        text = re.sub(r'\[x\]', '<gap>', text)
        text = re.sub(r'\.\.\.', '<big_gap>', text)
        text = re.sub(r'\[\.\.\.\s*\.\.\.\]', '<big_gap>', text)
        text = re.sub(r'\[…\s*…\]', '<big_gap>', text)
        
        # Normalize subscripts (e.g., il₅ -> il5)
        text = self._normalize_subscripts(text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _normalize_subscripts(self, text: str) -> str:
        """Convert subscript numbers to regular numbers"""
        subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
        text = text.translate(subscript_map)
        text = text.translate(superscript_map)
        return text
    
    def clean_translation(self, text: str) -> str:
        """Clean English translation text"""
        
        # Remove line numbers (1, 5, 10, 15', 20'', etc.)
        text = re.sub(r'\b\d+[\'\"]*\s*', '', text)
        
        # Remove parenthetical comments about breaks/erasures
        text = re.sub(r'\([^)]*break[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*erasure[^)]*\)', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def segment_words(self, akkadian_text: str) -> List[str]:
        """
        Segment Akkadian text into words.
        Akkadian uses hyphens within words (syllabic writing)
        """
        # Split on spaces, keep hyphenated syllables together
        words = akkadian_text.split()
        return words
    
    def prepare_pair(self, source: str, target: str) -> Tuple[str, str]:
        """Prepare a source-target pair for training"""
        source = self.clean_transliteration(source)
        target = self.clean_translation(target)
        return source, target
```

### 3.2 Handling Proper Nouns

```python
class ProperNounHandler:
    """
    Handle proper nouns which are critical for BLEU score.
    Capitalized first letter = proper noun
    ALL CAPS = Sumerian logogram
    """
    
    def __init__(self, lexicon_path: str = None):
        self.proper_nouns = set()
        self.logograms = {}
        
        if lexicon_path:
            self.load_lexicon(lexicon_path)
    
    def load_lexicon(self, path: str):
        """Load the provided lexicon of proper nouns"""
        import pandas as pd
        df = pd.read_csv(path)
        # Assuming lexicon has columns like 'akkadian', 'english'
        for _, row in df.iterrows():
            self.proper_nouns.add(row['akkadian'].lower())
    
    def is_proper_noun(self, word: str) -> bool:
        """Check if word is a proper noun (first letter capitalized)"""
        if not word:
            return False
        return word[0].isupper() and not word.isupper()
    
    def is_logogram(self, word: str) -> bool:
        """Check if word is a Sumerian logogram (ALL CAPS)"""
        if not word:
            return False
        # Remove determinatives and check
        clean = re.sub(r'\{[^}]+\}', '', word)
        return clean.isupper() and len(clean) > 1
    
    def add_special_tokens(self, text: str) -> str:
        """Add special tokens to mark proper nouns and logograms"""
        words = text.split()
        result = []
        
        for word in words:
            if self.is_logogram(word):
                result.append(f'<LOG>{word}</LOG>')
            elif self.is_proper_noun(word):
                result.append(f'<PN>{word}</PN>')
            else:
                result.append(word)
        
        return ' '.join(result)
```

### 3.3 Data Augmentation for Low-Resource NMT

```python
import random

class DataAugmenter:
    """
    Data augmentation techniques for low-resource NMT
    """
    
    def __init__(self, noise_prob: float = 0.1):
        self.noise_prob = noise_prob
    
    def shuffle_within_window(self, tokens: List[str], window: int = 3) -> List[str]:
        """Shuffle tokens within small windows (helps with word order flexibility)"""
        result = []
        for i in range(0, len(tokens), window):
            chunk = tokens[i:i+window]
            random.shuffle(chunk)
            result.extend(chunk)
        return result
    
    def random_deletion(self, tokens: List[str], prob: float = 0.1) -> List[str]:
        """Randomly delete tokens"""
        return [t for t in tokens if random.random() > prob]
    
    def synonym_replacement(self, tokens: List[str], synonym_dict: dict) -> List[str]:
        """Replace tokens with synonyms from dictionary"""
        result = []
        for token in tokens:
            if token in synonym_dict and random.random() < 0.1:
                result.append(random.choice(synonym_dict[token]))
            else:
                result.append(token)
        return result
    
    def backtranslation_placeholder(self, source: str, target: str) -> Tuple[str, str]:
        """
        Placeholder for backtranslation augmentation.
        In practice, you'd:
        1. Train English->Akkadian model
        2. Translate English to synthetic Akkadian
        3. Use (synthetic_akkadian, original_english) as new training pair
        """
        # This would require a trained reverse model
        pass
    
    def augment_pair(self, source: str, target: str) -> List[Tuple[str, str]]:
        """Generate augmented training pairs"""
        pairs = [(source, target)]  # Original
        
        source_tokens = source.split()
        
        # Augmentation 1: Minor noise in source
        if len(source_tokens) > 3:
            noisy = ' '.join(self.random_deletion(source_tokens, 0.05))
            pairs.append((noisy, target))
        
        return pairs
```

---

## 4. Model Architecture Options

### Option 1: Fine-tune Pretrained Multilingual Model (Recommended)

This is your best bet for a low-resource language.

```python
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

class ModelFactory:
    """Factory for creating translation models"""
    
    @staticmethod
    def get_mbart(model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        """
        mBART: Multilingual BART
        - Pretrained on 50 languages
        - Good for low-resource transfer
        """
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        
        # Set source and target languages
        # Akkadian is not in mBART, so we'll use a similar Semitic language placeholder
        tokenizer.src_lang = "ar_AR"  # Arabic as proxy
        tokenizer.tgt_lang = "en_XX"
        
        return model, tokenizer
    
    @staticmethod
    def get_mt5(model_name: str = "google/mt5-base"):
        """
        mT5: Multilingual T5
        - Pretrained on 101 languages
        - More flexible with new languages
        """
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    
    @staticmethod
    def get_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
        """
        NLLB (No Language Left Behind)
        - Designed for low-resource languages
        - 200+ languages
        - RECOMMENDED for this challenge
        """
        from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
        
        tokenizer = NllbTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
```

### Option 2: Custom Transformer (If You Want Full Control)

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AkkadianTransformer(nn.Module):
    """
    Custom Transformer for Akkadian-English translation
    Optimized for low-resource scenario
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,  # Higher dropout for low-resource
        max_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        src_padding_mask: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
```

---

## 5. Training Strategy

### 5.1 Tokenization Strategy

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

def train_bpe_tokenizer(texts: List[str], vocab_size: int = 8000, save_path: str = "tokenizer.json"):
    """
    Train a BPE tokenizer specifically for Akkadian
    BPE is better than word-level for morphologically rich languages
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer: split on whitespace but keep hyphens
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<gap>", "<big_gap>"],
        min_frequency=2,
        show_progress=True
    )
    
    # Train
    tokenizer.train_from_iterator(texts, trainer)
    
    # Add post-processor for BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", 2), ("<eos>", 3)]
    )
    
    tokenizer.save(save_path)
    return tokenizer


def train_sentencepiece(texts: List[str], prefix: str = "akkadian", vocab_size: int = 8000):
    """
    Alternative: SentencePiece (unigram or BPE)
    Often works better for rare languages
    """
    import sentencepiece as spm
    
    # Write texts to file
    with open("train_texts.txt", "w") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Train
    spm.SentencePieceTrainer.train(
        input="train_texts.txt",
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="unigram",  # or "bpe"
        character_coverage=0.9995,
        user_defined_symbols=["<gap>", "<big_gap>"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    
    return spm.SentencePieceProcessor(model_file=f"{prefix}.model")
```

### 5.2 Training Loop with HuggingFace

```python
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np

def prepare_dataset(df, tokenizer, max_source_length=128, max_target_length=128):
    """Prepare dataset for training"""
    
    def preprocess_function(examples):
        inputs = examples['source']
        targets = examples['target']
        
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding=False
        )
        
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding=False
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    dataset = Dataset.from_pandas(df)
    tokenized = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    return tokenized


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU and chrF++ for evaluation"""
    import sacrebleu
    
    preds, labels = eval_preds
    
    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 with pad token id
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute scores
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
    
    geo_mean = (bleu.score * chrf.score) ** 0.5
    
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "geo_mean": geo_mean
    }


def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./model"):
    """Main training function"""
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
        warmup_steps=500,
        fp16=True,  # Mixed precision
        gradient_accumulation_steps=2,
        label_smoothing_factor=0.1,  # Helps with low-resource
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    trainer.train()
    return trainer
```

### 5.3 Low-Resource Training Tips

```python
class LowResourceTrainer:
    """
    Special techniques for low-resource NMT
    """
    
    @staticmethod
    def gradual_unfreezing(model, tokenizer, train_data, num_stages=3):
        """
        Gradually unfreeze layers during training
        Start with only the output layer, then progressively unfreeze more
        """
        # Stage 1: Freeze all except output layer
        for name, param in model.named_parameters():
            if 'lm_head' not in name and 'fc_out' not in name:
                param.requires_grad = False
        
        # Train stage 1
        # ... (train for a few epochs)
        
        # Stage 2: Unfreeze decoder
        for name, param in model.named_parameters():
            if 'decoder' in name:
                param.requires_grad = True
        
        # Train stage 2
        # ...
        
        # Stage 3: Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
        
        # Final training
        # ...
    
    @staticmethod
    def r_drop_loss(model, batch, alpha=0.5):
        """
        R-Drop: Regularized Dropout
        Pass input through model twice with different dropout masks
        Add KL divergence between the two outputs
        """
        import torch.nn.functional as F
        
        # Forward pass 1
        output1 = model(**batch)
        logits1 = output1.logits
        
        # Forward pass 2 (different dropout)
        output2 = model(**batch)
        logits2 = output2.logits
        
        # Standard CE loss
        ce_loss = (output1.loss + output2.loss) / 2
        
        # KL divergence between two passes
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        kl_loss = F.kl_div(p, q, log_target=True, reduction='batchmean')
        
        return ce_loss + alpha * kl_loss
    
    @staticmethod
    def curriculum_learning(train_data, model, tokenizer, num_stages=3):
        """
        Start with easy examples, gradually increase difficulty
        Easy = shorter sequences, more common words
        """
        # Sort by length
        sorted_data = sorted(train_data, key=lambda x: len(x['source'].split()))
        
        chunk_size = len(sorted_data) // num_stages
        
        for stage in range(num_stages):
            stage_data = sorted_data[:chunk_size * (stage + 1)]
            # Train on progressively larger/harder data
            # ...
```

---

## 6. Inference & Submission

### 6.1 Beam Search with Constraints

```python
def generate_translations(model, tokenizer, test_texts, batch_size=32):
    """
    Generate translations with beam search
    """
    model.eval()
    translations = []
    
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"] if hasattr(tokenizer, 'lang_code_to_id') else None
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    
    return translations


def post_process_translation(text: str) -> str:
    """Post-process translation for submission"""
    # Ensure single sentence (no newlines)
    text = text.replace('\n', ' ')
    
    # Fix common issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    text = text.strip()
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Ensure ends with period if it doesn't have punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text
```

### 6.2 Ensemble Methods

```python
def ensemble_predictions(models, tokenizers, test_texts):
    """
    Combine predictions from multiple models
    """
    all_translations = []
    
    for model, tokenizer in zip(models, tokenizers):
        translations = generate_translations(model, tokenizer, test_texts)
        all_translations.append(translations)
    
    # Simple voting: pick most common translation
    # Or: use model confidence scores
    
    final_translations = []
    for i in range(len(test_texts)):
        candidates = [t[i] for t in all_translations]
        # Pick most common (or use scoring)
        from collections import Counter
        most_common = Counter(candidates).most_common(1)[0][0]
        final_translations.append(most_common)
    
    return final_translations
```

### 6.3 Create Submission

```python
import pandas as pd

def create_submission(test_df, translations, output_path="submission.csv"):
    """Create submission file"""
    
    # Post-process all translations
    processed = [post_process_translation(t) for t in translations]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_df['id'],
        'translation': processed
    })
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Sample:\n{submission.head()}")
    
    return submission
```

---

## 7. Complete Code Implementation

### Full Kaggle Notebook Template

```python
# ============================================
# DEEP PAST CHALLENGE - AKKADIAN TO ENGLISH
# ============================================

import os
import re
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import sacrebleu
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CONFIGURATION
# ============================================

class Config:
    # Paths
    TRAIN_PATH = "/kaggle/input/deep-past-challenge/train.csv"
    TEST_PATH = "/kaggle/input/deep-past-challenge/test.csv"
    LEXICON_PATH = "/kaggle/input/deep-past-challenge/lexicon.csv"
    OUTPUT_DIR = "/kaggle/working/model"
    
    # Model
    MODEL_NAME = "facebook/nllb-200-distilled-600M"  # Best for low-resource
    # Alternative: "google/mt5-small"
    
    # Training
    MAX_SOURCE_LENGTH = 128
    MAX_TARGET_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 500
    
    # Inference
    NUM_BEAMS = 5
    MAX_GEN_LENGTH = 150


# ============================================
# 2. PREPROCESSING
# ============================================

def clean_akkadian(text):
    """Clean Akkadian transliteration"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove modern notations
    text = re.sub(r'[!?/]', '', text)
    text = re.sub(r'[:\.]', ' ', text)
    
    # Handle brackets
    text = re.sub(r'<([^>]+)>', r'\1', text)
    text = re.sub(r'<<[^>]+>>', '', text)
    text = re.sub(r'[˹˺]', '', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    
    # Handle gaps
    text = re.sub(r'\[x\]', '<gap>', text)
    text = re.sub(r'\.\.\.+', '<big_gap>', text)
    
    # Normalize subscripts
    subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    text = text.translate(subscript_map)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def clean_english(text):
    """Clean English translation"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove line numbers
    text = re.sub(r'\b\d+[\'\"]*\s*', '', text)
    
    # Normalize
    text = ' '.join(text.split())
    
    return text


# ============================================
# 3. DATASET PREPARATION
# ============================================

def prepare_data():
    """Load and prepare data"""
    
    # Load data
    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Clean
    train_df['source_clean'] = train_df['transliteration'].apply(clean_akkadian)
    train_df['target_clean'] = train_df['translation'].apply(clean_english)
    
    test_df['source_clean'] = test_df['transliteration'].apply(clean_akkadian)
    
    # Remove empty rows
    train_df = train_df[
        (train_df['source_clean'].str.len() > 0) & 
        (train_df['target_clean'].str.len() > 0)
    ].reset_index(drop=True)
    
    print(f"Train size after cleaning: {len(train_df)}")
    
    return train_df, test_df


def tokenize_data(train_df, tokenizer):
    """Tokenize data for training"""
    
    def preprocess(examples):
        inputs = examples['source_clean']
        targets = examples['target_clean']
        
        model_inputs = tokenizer(
            inputs,
            max_length=Config.MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False
        )
        
        labels = tokenizer(
            targets,
            max_length=Config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Split train/val
    train_size = int(0.95 * len(train_df))
    train_data = train_df[:train_size]
    val_data = train_df[train_size:]
    
    train_dataset = Dataset.from_pandas(train_data[['source_clean', 'target_clean']])
    val_dataset = Dataset.from_pandas(val_data[['source_clean', 'target_clean']])
    
    train_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=['source_clean', 'target_clean'])
    val_tokenized = val_dataset.map(preprocess, batched=True, remove_columns=['source_clean', 'target_clean'])
    
    return train_tokenized, val_tokenized


# ============================================
# 4. MODEL & TRAINING
# ============================================

def compute_metrics(eval_preds, tokenizer):
    """Compute evaluation metrics"""
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
    
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "geo_mean": (bleu.score * chrf.score) ** 0.5
    }


def train():
    """Main training function"""
    
    # Prepare data
    train_df, test_df = prepare_data()
    
    # Load model and tokenizer
    print(f"Loading model: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
    
    # Tokenize
    train_dataset, val_dataset = tokenize_data(train_df, tokenizer)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=Config.EPOCHS,
        predict_with_generate=True,
        generation_max_length=Config.MAX_GEN_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
        warmup_steps=Config.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        label_smoothing_factor=0.1,
        logging_steps=100,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    return model, tokenizer, test_df


# ============================================
# 5. INFERENCE
# ============================================

def generate_translations(model, tokenizer, texts):
    """Generate translations"""
    model.eval()
    device = next(model.parameters()).device
    
    translations = []
    batch_size = Config.BATCH_SIZE
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_SOURCE_LENGTH
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=Config.MAX_GEN_LENGTH,
                num_beams=Config.NUM_BEAMS,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    
    return translations


def post_process(text):
    """Post-process translation"""
    text = str(text).replace('\n', ' ')
    text = ' '.join(text.split())
    text = text.strip()
    
    if text and text[-1] not in '.!?':
        text += '.'
    
    if text:
        text = text[0].upper() + text[1:]
    
    return text


def create_submission(test_df, translations):
    """Create submission file"""
    processed = [post_process(t) for t in translations]
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'translation': processed
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved!")
    print(submission.head())


# ============================================
# 6. MAIN
# ============================================

if __name__ == "__main__":
    # Train
    model, tokenizer, test_df = train()
    
    # Generate
    test_texts = test_df['source_clean'].tolist()
    translations = generate_translations(model, tokenizer, test_texts)
    
    # Submit
    create_submission(test_df, translations)
```

---

## 8. Tips for Winning

### Data-Related
1. **Study the lexicon**: Map proper nouns exactly
2. **Handle determinatives**: `{ki}`, `{d}` etc. should be consistent
3. **Preserve logograms**: ALL CAPS words are Sumerian → handle specially
4. **Data augmentation**: Back-translation if possible

### Model-Related
1. **Use NLLB or mT5**: Best for low-resource languages
2. **Try smaller models first**: They train faster, iterate quicker
3. **Ensemble 2-3 models**: Different architectures/seeds
4. **Label smoothing**: 0.1-0.2 helps generalization

### Training-Related
1. **More epochs, small LR**: Low-resource = need more iterations
2. **Larger batch (gradient accumulation)**: Stabilizes training
3. **Early stopping**: Prevent overfitting
4. **Curriculum learning**: Start with shorter sequences

### Inference-Related
1. **Beam search > greedy**: num_beams=5 is a good start
2. **Length penalty tuning**: If translations too short/long
3. **Post-processing matters**: Proper capitalization, punctuation

### Competition-Specific
1. **Read the discussions**: Others share insights
2. **Look at public notebooks**: Learn what works
3. **Check your submission format**: Common cause of errors
4. **Save runtime for inference**: 9-hour limit is strict

---

## Quick Start Commands

```bash
# Install dependencies (in Kaggle)
!pip install sacrebleu sentencepiece transformers datasets

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

Good luck with the challenge, Edwin! 🏆 This is a fascinating problem that combines NLP, history, and archaeology. Let me know if you need clarification on any section!
