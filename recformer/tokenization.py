import torch
from transformers import BertTokenizerFast

class RecformerTokenizer(BertTokenizerFast):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None):
        cls.config = config
        return super().from_pretrained(pretrained_model_name_or_path)
        
    def __call__(self, items, pad_to_max=False, return_tensor=False):
        '''
        items: item sequence or a batch of item sequence, item sequence is a list of dict of item attributes
        
        return:
        input_ids: sequence of token ids
        item_position_ids: sequence of item positions
        token_type_ids: sequence of token types (cls, key, value, or padding)
        attention_mask: local attention masks
        global_attention_mask: global attention masks for Longformer
        '''
        # Is batch or not
        inputDict = self.batch_encode(items, pad_to_max=pad_to_max) \
            if type(items[0])==list and len(items)>0 \
            else self.encode(items)
        # Convert list to long tensor
        if return_tensor:
            for key, val in inputDict.items(): inputDict[key]=torch.LongTensor(val)
        return inputDict

    def tokenize_txt(self, txt):
        '''
        Tokenize a text.
        '''
        tokenized_text=self.tokenize(txt)
        idsOfText=self.convert_tokens_to_ids(tokenized_text)
        return idsOfText

    def encode_item(self, itemDict: dict):
        '''
        Encode an item by its attributes.
        '''
        item_token_ids, token_type_ids = [], []
        for attr_idx, attr_tuple in enumerate(itemDict.items(), start=0):
            if attr_idx>=self.config.max_attr_num: break # max number of attributes allowed
            attr_key, attr_val = attr_tuple
            key_token_ids, val_token_ids = self.tokenize_txt(attr_key), self.tokenize_txt(attr_val)
            attr_token_ids = (key_token_ids + val_token_ids)[:self.config.max_attr_length] # max number of tokens allowed
            item_token_ids.extend(attr_token_ids) # add attribute tokens to item encoding
            attr_type_ids = ([1]*len(key_token_ids) + [2]*len(val_token_ids))[:self.config.max_attr_length] # 1: attribute key, 2: attribute value, still truncate
            token_type_ids.extend(attr_type_ids) # record the corresponding types
        return item_token_ids, token_type_ids

    def encode(self, itemSeq, encode_item=True):
        '''
        Encode an item sequence in reverse order of time, assuming the sequence is in increasing order of time.
        '''
        itemSeq = itemSeq[::-1][:self.config.max_item_embeddings - 1]  # reverse order of time, truncate max number of items, leave one space for <cls>
        input_ids, position_ids, token_type_ids = [self.cls_token_id], [0], [0] # input tokens, position tokens, type tokens (init for <cls>)
        for idx, itemInfo in enumerate(itemSeq, start=1):
            item_token_ids, item_token_type_ids = self.encode_item(itemInfo) if encode_item else itemInfo # obtain token ids and corresponding types
            input_ids.extend(item_token_ids); position_ids.extend([idx]*len(item_token_ids)); token_type_ids.extend(item_token_type_ids) # update input tokens, position tokens, type tokens

        input_ids, position_ids, token_type_ids = input_ids[:self.config.max_token_num], position_ids[:self.config.max_token_num], token_type_ids[:self.config.max_token_num] # Truncate by max_token_num
        attention_mask, global_attention_mask = [1]*len(input_ids), [1]+[0]*(len(input_ids)-1)
        return {
            "input_ids": input_ids, "item_position_ids": position_ids, "token_type_ids": token_type_ids,
            "attention_mask": attention_mask, "global_attention_mask": global_attention_mask
            }

    def padding(self, batch, pad_to_max):
        '''
        Padding samples in a batch to align sequence lengths.
        '''
        max_len = self.config.max_token_num if pad_to_max else max([len(sample["input_ids"]) for sample in batch])
        batch_input_ids, batch_position_ids, batch_token_type_ids, batch_attention_mask, batch_global_attention_mask = [], [], [], [], []

        for sample in batch:
            input_ids, position_ids, token_type_ids, attention_mask, global_attention_mask = \
                sample["input_ids"], sample["item_position_ids"], sample["token_type_ids"], sample["attention_mask"], sample["global_attention_mask"]

            pad_length=max_len-len(input_ids)

            # Add padded sequences to batch lists
            input_ids.extend([self.pad_token_id]*pad_length); batch_input_ids.append(input_ids) # Pad token filling
            position_ids.extend([self.config.max_item_embeddings-1]*pad_length); batch_position_ids.append(position_ids) # Last position id
            token_type_ids.extend([3]*pad_length); batch_token_type_ids.append(token_type_ids) # Pad token type=3
            attention_mask.extend([0]*pad_length); batch_attention_mask.append(attention_mask) # No attention
            global_attention_mask.extend([0]*pad_length); batch_global_attention_mask.append(global_attention_mask) # No attention

        return {
            "input_ids": batch_input_ids, "item_position_ids": batch_position_ids, "token_type_ids": batch_token_type_ids,
            "attention_mask": batch_attention_mask, "global_attention_mask": batch_global_attention_mask
        }

    def batch_encode(self, batch, encode_item=True, pad_to_max=False):
        '''
        Encode a batch of item sequences and pad to equal length.
        '''
        batch = [self.encode(sample, encode_item) for sample in batch]
        padded_batch = self.padding(batch, pad_to_max)
        return padded_batch