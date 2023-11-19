import torch
from collections import OrderedDict
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

PRETRAINED_CKPT_PATH = 'result/recformer_pretraining/lightning_logs/version_5/checkpoints/epoch=0-avg_val_accuracy=0.1368.ckpt/pytorch_model.bin'
LONGFORMER_CKPT_PATH = 'longformer_ckpt/longformer-mini-1024.bin'
LONGFORMER_TYPE = 'kiddothe2b/longformer-mini-1024'
RECFORMERSEQREC_OUTPUT_PATH = 'pretrain_ckpt/seqrec_pretrain_ckpt-v5-epoch=0-avg_val_accuracy=0.1368.bin'

input_file = PRETRAINED_CKPT_PATH
state_dict = torch.load(input_file)

longformer_file = LONGFORMER_CKPT_PATH
longformer_state_dict = torch.load(longformer_file)

state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'] = longformer_state_dict['longformer.embeddings.word_embeddings.weight']

output_file = RECFORMERSEQREC_OUTPUT_PATH
new_state_dict = OrderedDict()

for key, value in state_dict.items():
    if key.startswith('_forward_module.model.'):
        new_key = key[len('_forward_module.model.'):]
        new_state_dict[new_key] = value

config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 6
model = RecformerForSeqRec(config)

model.load_state_dict(new_state_dict, strict=False)
torch.save(new_state_dict, output_file)

print("done")