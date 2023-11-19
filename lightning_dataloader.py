from collator import PretrainDataCollatorWithPadding
from torch.utils.data import Dataset
from typing import List

class ClickDataset(Dataset):
    def __init__(self, pretrainData: List, collator: PretrainDataCollatorWithPadding):
        super().__init__()
        self.pretrainData, self.collator = pretrainData, collator

    def __getitem__(self, idx): return self.pretrainData[idx]
    
    def __len__(self): return len(self.pretrainData)

    def collate_fn(self, batch): return self.collator([{'items': sample} for sample in batch])
