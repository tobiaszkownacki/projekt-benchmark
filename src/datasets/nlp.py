"""Natural Language Processing datasets"""

import torch
from torch.utils.data import ConcatDataset, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import RAW_DATA_DIR
from src.datasets.base import BaseDatasetFactory


class NLPDatasetFactory(BaseDatasetFactory):
    @classmethod
    def get_data_set(cls, dataset_name: str) -> ConcatDataset:
        match dataset_name:
            case "wmt14":
                return cls._get_wmt14_data_set()
            case "imdb":
                return cls._get_imdb_data_set()
            case _:
                raise ValueError(f"Unsupported NLP dataset: {dataset_name}")
    
    @classmethod
    def get_supported_datasets(cls) -> list[str]:
        return ["wmt14", "imdb"]
    
    @classmethod
    def _get_wmt14_data_set(
        cls, 
        train_size: int = 10000,
        val_size: int = 1000,
        max_length: int = 128
    ) -> ConcatDataset:
        cache_dir = RAW_DATA_DIR / "wmt14"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading WMT14 dataset (train={train_size}, val={val_size})...")
        ds = load_dataset(
            "wmt/wmt14", 
            "de-en",
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        
        train_ds = ds['train'].select(range(train_size))
        val_ds = ds['validation'].select(range(val_size))
        
        print(f"Example: {train_ds[0]['translation']}")
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        
        def tokenize_batch(examples):
            de_texts = [ex['de'] for ex in examples['translation']]
            en_texts = [ex['en'] for ex in examples['translation']]
            
            de_tokens = tokenizer(
                de_texts, 
                padding='max_length', 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt'
            )
            en_tokens = tokenizer(
                en_texts, 
                padding='max_length',
                truncation=True, 
                max_length=max_length,
                return_tensors='pt'
            )
            
            return de_tokens['input_ids'], en_tokens['input_ids']
        
        print("Tokenizing train set...")
        train_de_ids = []
        train_en_ids = []
        
        for i in tqdm(range(0, len(train_ds), 100), desc="Train"):
            batch = train_ds[i:min(i+100, len(train_ds))]
            de_ids, en_ids = tokenize_batch(batch)
            train_de_ids.append(de_ids)
            train_en_ids.append(en_ids)
        
        train_de_tensor = torch.cat(train_de_ids, dim=0)
        train_en_tensor = torch.cat(train_en_ids, dim=0)
        
        
        train_dataset = TensorDataset(train_de_tensor, train_en_tensor)
        
        return train_dataset
    
    @classmethod
    def _get_imdb_data_set(
        cls,
        max_samples: int = 5000,
        max_length: int = 256
    ) -> ConcatDataset:
        """IMDB sentiment classification dataset"""
        cache_dir = RAW_DATA_DIR / "imdb"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading IMDB dataset (max={max_samples})...")
        ds = load_dataset("imdb", cache_dir=str(cache_dir))
        
        train_ds = ds['train'].select(range(max_samples))
        val_ds = ds['test'].select(range(max_samples // 5))
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        def tokenize_batch(texts, labels):
            tokens = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            return tokens['input_ids'], torch.tensor(labels, dtype=torch.long)
        
        print("Tokenizing IMDB...")
        train_ids, train_labels = tokenize_batch(
            train_ds['text'], 
            train_ds['label']
        )
        val_ids, val_labels = tokenize_batch(
            val_ds['text'],
            val_ds['label']
        )
        
        train_dataset = TensorDataset(train_ids, train_labels)
        val_dataset = TensorDataset(val_ids, val_labels)
        
        return ConcatDataset([train_dataset, val_dataset])