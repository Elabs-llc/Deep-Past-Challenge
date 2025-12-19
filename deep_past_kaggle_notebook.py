# Deep Past Challenge - Akkadian to English Translation
# Kaggle Competition Notebook

"""
This notebook provides a complete solution for the Deep Past Challenge.
It uses NLLB (No Language Left Behind) model which is optimized for low-resource languages.

To use:
1. Add this as a Kaggle notebook
2. Attach the competition data
3. Enable GPU accelerator
4. Run all cells
"""

# ============================================
# CELL 1: IMPORTS & SETUP
# ============================================

import os
import re
import gc
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Install required packages
!pip install -q sacrebleu sentencepiece transformers datasets accelerate

# ============================================
# CELL 2: CONFIGURATION
# ============================================

class Config:
    """All configuration in one place"""
    
    # Paths - UPDATE THESE FOR YOUR KAGGLE ENVIRONMENT
    DATA_DIR = "/kaggle/input/deep-past-challenge"
    TRAIN_PATH = f"{DATA_DIR}/train.csv"
    TEST_PATH = f"{DATA_DIR}/test.csv"
    OUTPUT_DIR = "/kaggle/working/checkpoints"
    
    # Model options (uncomment one):
    # Option 1: NLLB - Best for low-resource (RECOMMENDED)
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    
    # Option 2: mT5 - Good alternative
    # MODEL_NAME = "google/mt5-small"
    
    # Option 3: mBART - Another option
    # MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
    
    # Data
    MAX_SOURCE_LEN = 128
    MAX_TARGET_LEN = 128
    VAL_SPLIT = 0.05  # 5% for validation
    
    # Training
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2  # Effective batch = 32
    EPOCHS = 10
    LR = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    LABEL_SMOOTHING = 0.1
    
    # Inference
    NUM_BEAMS = 5
    MAX_GEN_LEN = 150
    LENGTH_PENALTY = 1.0
    
    # Misc
    SEED = 42
    
    @classmethod
    def display(cls):
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print("=" * 50)

Config.display()

# ============================================
# CELL 3: DATA PREPROCESSING
# ============================================

class AkkadianPreprocessor:
    """
    Preprocesses Akkadian transliterations and English translations
    following competition guidelines.
    """
    
    # Unicode subscripts to regular numbers
    SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    SUPERSCRIPT_MAP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    
    @staticmethod
    def clean_akkadian(text: str) -> str:
        """Clean Akkadian transliteration text"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        
        # Step 1: Remove modern scribal notations
        text = re.sub(r'!', '', text)   # Certain reading
        text = re.sub(r'\?', '', text)  # Uncertain reading
        text = re.sub(r'/', ' ', text)  # Line divider -> space
        text = re.sub(r'[:\.]', ' ', text)  # Word dividers -> space
        
        # Step 2: Handle brackets (keep content, remove brackets)
        text = re.sub(r'<([^>]*)>', r'\1', text)     # Scribal insertions
        text = re.sub(r'<<[^>]*>>', '', text)        # Erroneous signs (remove entirely)
        text = re.sub(r'[˹˺]', '', text)             # Partial break markers
        text = re.sub(r'\[([^\]]*)\]', r'\1', text)  # Square brackets
        
        # Step 3: Handle gaps and breaks
        text = re.sub(r'\[x\]', ' <gap> ', text)
        text = re.sub(r'\.\.\.+', ' <big_gap> ', text)
        text = re.sub(r'\[…+\s*…*\]', ' <big_gap> ', text)
        text = re.sub(r'…+', ' <big_gap> ', text)
        
        # Step 4: Normalize subscripts and superscripts
        text = text.translate(AkkadianPreprocessor.SUBSCRIPT_MAP)
        text = text.translate(AkkadianPreprocessor.SUPERSCRIPT_MAP)
        
        # Step 5: Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def clean_english(text: str) -> str:
        """Clean English translation text"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        
        # Remove line numbers (1, 5, 10', 15'', etc.)
        text = re.sub(r"^\d+['\"]*\s*", '', text)  # At start
        text = re.sub(r"\s+\d+['\"]*\s+", ' ', text)  # In middle
        
        # Remove parenthetical comments about breaks/erasures
        text = re.sub(r'\([^)]*break[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*erasure[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*lacuna[^)]*\)', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def post_process_translation(text: str) -> str:
        """Post-process generated translation for submission"""
        if not text:
            return "Translation unavailable."
        
        text = str(text)
        
        # Ensure single line
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text.strip()


# Test preprocessing
test_akk = "a-na A-šur-{d}UTU [x] ša KÙ.BABBAR..."
test_eng = "1 To Assur-Shamash, [break] who (has) the silver..."

print("Original Akkadian:", test_akk)
print("Cleaned Akkadian:", AkkadianPreprocessor.clean_akkadian(test_akk))
print()
print("Original English:", test_eng)
print("Cleaned English:", AkkadianPreprocessor.clean_english(test_eng))


# ============================================
# CELL 4: LOAD AND PREPARE DATA
# ============================================

def load_and_prepare_data():
    """Load competition data and prepare for training"""
    
    # Load CSVs
    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    
    print(f"Original train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Show sample
    print("\nSample training data:")
    print(train_df.head(2))
    
    # Identify column names (may vary)
    # Common patterns: 'transliteration', 'akkadian', 'source' for source
    # 'translation', 'english', 'target' for target
    
    source_col = None
    target_col = None
    
    for col in train_df.columns:
        if 'translit' in col.lower() or 'akkadian' in col.lower() or 'source' in col.lower():
            source_col = col
        if 'translat' in col.lower() or 'english' in col.lower() or 'target' in col.lower():
            target_col = col
    
    print(f"\nIdentified source column: {source_col}")
    print(f"Identified target column: {target_col}")
    
    # Apply preprocessing
    train_df['source_clean'] = train_df[source_col].apply(AkkadianPreprocessor.clean_akkadian)
    train_df['target_clean'] = train_df[target_col].apply(AkkadianPreprocessor.clean_english)
    
    test_df['source_clean'] = test_df[source_col].apply(AkkadianPreprocessor.clean_akkadian)
    
    # Remove empty rows
    before = len(train_df)
    train_df = train_df[
        (train_df['source_clean'].str.len() > 5) & 
        (train_df['target_clean'].str.len() > 5)
    ].reset_index(drop=True)
    
    print(f"\nRemoved {before - len(train_df)} empty/short rows")
    print(f"Final train size: {len(train_df)}")
    
    # Show cleaned sample
    print("\nCleaned sample:")
    print(f"Source: {train_df['source_clean'].iloc[0][:100]}...")
    print(f"Target: {train_df['target_clean'].iloc[0][:100]}...")
    
    return train_df, test_df


train_df, test_df = load_and_prepare_data()


# ============================================
# CELL 5: SETUP MODEL AND TOKENIZER
# ============================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def setup_model():
    """Load model and tokenizer"""
    
    print(f"Loading model: {Config.MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME)
    
    # Add special tokens for gaps
    special_tokens = ['<gap>', '<big_gap>']
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded on: {device}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


model, tokenizer, device = setup_model()


# ============================================
# CELL 6: CREATE DATASETS
# ============================================

from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

def create_datasets(train_df, tokenizer):
    """Create train and validation datasets"""
    
    # Shuffle and split
    train_df = train_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    val_size = int(len(train_df) * Config.VAL_SPLIT)
    val_df = train_df[:val_size]
    train_df = train_df[val_size:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    def preprocess(examples):
        inputs = tokenizer(
            examples['source_clean'],
            max_length=Config.MAX_SOURCE_LEN,
            truncation=True,
            padding='max_length'
        )
        
        targets = tokenizer(
            examples['target_clean'],
            max_length=Config.MAX_TARGET_LEN,
            truncation=True,
            padding='max_length'
        )
        
        # Replace padding token id with -100 for loss calculation
        labels = []
        for target_ids in targets['input_ids']:
            labels.append([
                -100 if token_id == tokenizer.pad_token_id else token_id 
                for token_id in target_ids
            ])
        
        inputs['labels'] = labels
        return inputs
    
    # Convert to HuggingFace datasets
    train_dataset = HFDataset.from_pandas(train_df[['source_clean', 'target_clean']])
    val_dataset = HFDataset.from_pandas(val_df[['source_clean', 'target_clean']])
    
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=['source_clean', 'target_clean'])
    val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=['source_clean', 'target_clean'])
    
    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    
    return train_dataset, val_dataset


train_dataset, val_dataset = create_datasets(train_df, tokenizer)
print(f"Train dataset features: {train_dataset.features}")


# ============================================
# CELL 7: TRAINING
# ============================================

from transformers import (
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import sacrebleu

def compute_metrics(eval_preds):
    """Compute BLEU and chrF++ metrics"""
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 with pad token
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute scores
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
    
    geo_mean = np.sqrt(bleu.score * chrf.score) if bleu.score > 0 and chrf.score > 0 else 0
    
    return {
        'bleu': round(bleu.score, 4),
        'chrf': round(chrf.score, 4),
        'geo_mean': round(geo_mean, 4)
    }


def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model"""
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        num_train_epochs=Config.EPOCHS,
        warmup_ratio=Config.WARMUP_RATIO,
        predict_with_generate=True,
        generation_max_length=Config.MAX_GEN_LEN,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        logging_steps=50,
        report_to="none",  # Disable wandb etc.
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    
    trainer.train()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    return trainer


# Run training
trainer = train_model(model, tokenizer, train_dataset, val_dataset)


# ============================================
# CELL 8: INFERENCE
# ============================================

def generate_translations(model, tokenizer, texts, batch_size=16):
    """Generate translations for test set"""
    
    model.eval()
    device = next(model.parameters()).device
    
    all_translations = []
    
    print(f"Generating translations for {len(texts)} samples...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            max_length=Config.MAX_SOURCE_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=Config.MAX_GEN_LEN,
                num_beams=Config.NUM_BEAMS,
                length_penalty=Config.LENGTH_PENALTY,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend(decoded)
        
        # Clear cache
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return all_translations


# Generate translations
test_texts = test_df['source_clean'].tolist()
translations = generate_translations(model, tokenizer, test_texts)

# Show samples
print("\nSample translations:")
for i in range(min(3, len(translations))):
    print(f"\nSource: {test_texts[i][:80]}...")
    print(f"Translation: {translations[i][:80]}...")


# ============================================
# CELL 9: CREATE SUBMISSION
# ============================================

def create_submission(test_df, translations):
    """Create submission file"""
    
    # Post-process translations
    processed = [
        AkkadianPreprocessor.post_process_translation(t) 
        for t in translations
    ]
    
    # Identify ID column
    id_col = 'id' if 'id' in test_df.columns else test_df.columns[0]
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df[id_col],
        'translation': processed
    })
    
    # Validate
    assert len(submission) == len(test_df), "Submission length mismatch!"
    assert submission['translation'].isna().sum() == 0, "Found NaN translations!"
    
    # Save
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "=" * 50)
    print("SUBMISSION CREATED")
    print("=" * 50)
    print(f"Shape: {submission.shape}")
    print(f"\nFirst 5 rows:")
    print(submission.head())
    print(f"\nLast 5 rows:")
    print(submission.tail())
    
    # Check file
    print(f"\nFile size: {os.path.getsize('submission.csv') / 1024:.2f} KB")
    
    return submission


submission = create_submission(test_df, translations)


# ============================================
# CELL 10: VALIDATION & CLEANUP
# ============================================

# Validate submission format
def validate_submission(filepath='submission.csv'):
    """Validate submission file format"""
    
    df = pd.read_csv(filepath)
    
    checks = {
        "Has 'id' column": 'id' in df.columns,
        "Has 'translation' column": 'translation' in df.columns,
        "No missing translations": df['translation'].isna().sum() == 0,
        "No empty translations": (df['translation'].str.len() > 0).all(),
        "All translations are strings": df['translation'].apply(lambda x: isinstance(x, str)).all()
    }
    
    print("\nSubmission Validation:")
    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ Submission is valid and ready to submit!")
    else:
        print("\n✗ Please fix the issues above before submitting.")
    
    return all_passed


validate_submission()

# Cleanup
gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("NOTEBOOK COMPLETE")
print("=" * 50)
print("Your submission.csv is ready!")
print("Good luck! 🏆")
