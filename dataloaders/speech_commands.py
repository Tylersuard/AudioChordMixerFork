import os
import math
import numpy as np
import logging
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset, DatasetDict
from pathlib import Path
import pytorch_lightning as pl
from .utils.bucket_sampler import BucketSampler
from .utils.collators import collate_batch_horizontal

class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        taskname,
        num_workers,
        batch_size,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.cache_dir = self.get_cache_dir()

    def prepare_data(self):
        self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset = self.process_dataset()
        # shuffled_train_dataset = dataset['train'].shuffle()
        # dataset['train'] = shuffled_train_dataset
        # smaller_train_dataset = dataset['train'].select(range(10000))
        # dataset['train'] = smaller_train_dataset

        # shuffled_test_dataset = dataset['test'].shuffle()
        # dataset['test'] = shuffled_test_dataset
        # smaller_test_dataset = dataset['test'].select(range(1000))
        # dataset['test'] = smaller_test_dataset

        # shuffled_val_dataset = dataset['validation'].shuffle()
        # dataset['validation'] = shuffled_val_dataset
        # smaller_val_dataset = dataset['validation'].select(range(1000))
        # dataset['validation'] = smaller_val_dataset


        dataset.set_format(type="torch", columns=["sequence", "label", "len"])

        self.train_dataset, self.test_dataset = (
            dataset["train"],
            dataset["test"],
        )

    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()
        
        dataset = load_dataset('speech_commands','v0.01')
        
# Reduce dataset size for faster iteration
        dataset = dataset.filter(lambda example: 1 <= example['label'] <= 3)

        def change_labels(example):
            if example['label'] == 3:
                example['label'] = 0
            return example

        # Apply the function to the 'train' and 'test' splits
        dataset['train'] = dataset['train'].map(change_labels)
        dataset['test'] = dataset['test'].map(change_labels)
        dataset['validation'] = dataset['validation'].map(change_labels)

        label_counter = Counter(dataset['train']['label'])

# Print the count of instances for each label
        for label, count in label_counter.items():
            print(f'Label {label}: {count} instances')

        def quantize(example):
            min_val, max_val = -1.0, 1.0
            num_bits = 8
            sequence = []
            for sample in example["audio"]["array"]:
              try:
                sample = float(sample)
                sample_normalized = (sample - min_val) / (max_val - min_val)
                sample_quantized = int(sample_normalized * (2**num_bits - 1))
                sequence.append(sample_quantized)
              except:
                print(sample)
            example["sequence"] = sequence
            example["len"] = len(sequence)
            return example
        
        dataset = dataset.map(
            quantize,
            remove_columns=["audio"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        
        self._save_to_cache(dataset)
        return dataset

    def _save_to_cache(self, dataset):
        cache_dir = self.data_dir / self._cache_dir_name
        os.makedirs(str(cache_dir), exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))

    def _load_from_cache(self):
        assert self.cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(self.cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(self.cache_dir))
        return dataset

    @property
    def _cache_dir_name(self):
        return f"speech_commands"

    def get_cache_dir(self):
        cache_dir = self.data_dir / self._cache_dir_name
        if cache_dir.is_dir():
            return cache_dir
        else:
            return None
        
    def get_bucket_boundaries(self, df):
        max_log2_bin = math.ceil(np.log2(max(df['len'])))
        min_log2_bin = math.floor(np.log2(min(df['len'])))
        return [2**i for i in range(min_log2_bin, max_log2_bin + 1)]
        
    def train_dataloader(self):
        boundaries = self.get_bucket_boundaries(self.train_dataset)
        sampler_train = BucketSampler(
            lengths=self.train_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler_train,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_batch_horizontal
        )
        return train_dataloader

    def val_dataloader(self):
        boundaries = self.get_bucket_boundaries(self.test_dataset)
        sampler_val = BucketSampler(
            lengths=self.test_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        val_dataloader = DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler_val,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_batch_horizontal
        )
        return val_dataloader

    def test_dataloader(self):
        boundaries = self.get_bucket_boundaries(self.test_dataset)
        sampler_test = BucketSampler(
            lengths=self.test_dataset['len'], 
            bucket_boundaries=boundaries,
            batch_size=self.batch_size
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler_test,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_batch_horizontal
        )
        return test_dataloader
