from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from torch.utils.data import Dataset

class RecformerTrainDataset(Dataset):
    def __init__(self, itemSeqOfUser:dict, collator:FinetuneDataCollatorWithPadding):
        self.itemSeqOfUser = itemSeqOfUser
        self.userIDs = list(itemSeqOfUser.keys()); self.userIDs.sort()
        self.collator = collator

    def __getitem__(self, idx): return self.itemSeqOfUser[self.userIDs[idx]]
    
    def __len__(self): return len(self.userIDs)

    def collate_fn(self, batch): return self.collator([{'items': sample} for sample in batch])

class RecformerEvalDataset(Dataset):
    def __init__(self, trainItemSeqOfUser, valItemSeqOfUser, testItemSeqOfUser, mode, collator: EvalDataCollatorWithPadding):
        self.trainItemSeqOfUser, self.valItemSeqOfUser, self.testItemSeqOfUser = trainItemSeqOfUser, valItemSeqOfUser, testItemSeqOfUser
        self.userIDs = list(self.valItemSeqOfUser.keys()) if mode == "val" else list(self.testItemSeqOfUser.keys())
        self.mode, self.collator = mode, collator

    def __getitem__(self, idx):
        userID = self.userIDs[idx]
        inputItemSeq = self.trainItemSeqOfUser[userID] if self.mode == "val" else self.trainItemSeqOfUser[userID]+self.valItemSeqOfUser[userID]
        gtNextItem = self.valItemSeqOfUser[userID] if self.mode == "val" else self.testItemSeqOfUser[userID]
        return inputItemSeq, gtNextItem
    
    def __len__(self): return len(self.userIDs)

    def collate_fn(self, batch): return self.collator([{'items': sample[0], 'label': sample[1]} for sample in batch])
