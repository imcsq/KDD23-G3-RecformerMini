import torch
torch.set_float32_matmul_precision('medium')

import logging
from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from transformers import LongformerForMaskedLM
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint 

from lighting_litwrapper import LitWrapper
from recformer import RecformerForPretraining, RecformerTokenizer, RecformerConfig
from collator import PretrainDataCollatorWithPadding
from lightning_dataloader import ClickDataset

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--dev_file', type=str, required=True)
parser.add_argument('--item_attr_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--dataloader_num_workers', type=int, default=2)
parser.add_argument('--mlm_probability', type=float, default=0.15)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--valid_step', type=int, default=2000)
parser.add_argument('--log_step', type=int, default=2000)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--fix_word_embedding', action='store_true')
parser.add_argument('--reverse_items', action='store_true')



tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    #  cy: may debug and have a look of the content of these values
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids


def main():
    
    args = parser.parse_args()
    print(args)
    seed_everything(42)
    # Sets a seed for random number generation to ensure reproducibility.

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51  # 50 item and 1 for cls
    config.attention_window = [64] * 6
    config.max_token_num = 512
    config.reverse_items = args.reverse_items
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    # 这里需要打个断点，看看cls的config

    global tokenizer_glb
    tokenizer_glb = tokenizer

    # preprocess corpus , use the BERT tokenizer to tokenize the corpus
    path_corpus = Path(args.item_attr_file) # [todo] debug: what is this file
    dir_corpus = path_corpus.parent
    dir_preprocess = dir_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'
    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        item_attrs = json.load(open(path_corpus))
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_attrs.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item_id: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        json.dump(tokenized_items, open(path_tokenized_items, 'w'))

    tokenized_items = json.load(open(path_tokenized_items))#dir_preprocess / f'attr_small.json'))#
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    # cy: loading data for pre-training, including the mlm_probability
    # data_collator: setting of data preprocessing
    data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mlm_probability=args.mlm_probability) 
    train_data = ClickDataset(json.load(open(args.train_file)), data_collator)
    dev_data = ClickDataset(json.load(open(args.dev_file)), data_collator)
    # format the training data OR dev_data??
    # ClickDataset will use the data_collator to process the data appropriately (like applying padding, token masking, etc.) for training.

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, # cy: what does this arguments mean?
                              shuffle=True, 
                              collate_fn=train_data.collate_fn,
                              num_workers=args.dataloader_num_workers)
    dev_loader = DataLoader(dev_data, 
                            batch_size=args.batch_size, 
                            collate_fn=dev_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
    # dev: development dataset == validation dataset

    # Load Longformer model to obtain the pretrained parameters
    Longformer = LongformerForMaskedLM.from_pretrained('kiddothe2b/longformer-mini-1024')
    Longformer_state_dict = Longformer.state_dict()

    # Build Recformer model to receive the pretrained parameters
    Recformer_config = RecformerConfig.from_pretrained('kiddothe2b/longformer-mini-1024')
    Recformer_config.max_attr_num = 3
    Recformer_config.max_attr_length = 32
    Recformer_config.max_item_embeddings = 51
    Recformer_config.attention_window = [64] * 6
    Recformer = RecformerForPretraining(Recformer_config)
    Recformer_state_dict = Recformer.state_dict()

    # Copy pretrained Longformer parameters to Recformer
    for lpname, lpval in Longformer_state_dict.items():
        if lpname not in Recformer_state_dict:
            print(f"Skip load unnecessary parameter {lpname}.")
            continue
        if lpval.size() != Recformer_state_dict[lpname].size():
            print(f"Skip load mismatched parameter {lpname}. Longformer size:", lpval.size(),
                  "Recformer expected size:", Recformer_state_dict[lpname].size())
            continue
        Recformer_state_dict[lpname].copy_(lpval)

    # mark: stop here. [3:29 pm]
    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in Recformer.longformer.embeddings.word_embeddings.parameters():
            # accesses the word embedding layer in the Longformer part of the Recformer.
            # prevents these embedding layers from being updated during training. to keep pre-trained embeddings intact from a well-trained model and only train the other parts of the model.
            # [?] why freeze the word embedding layer here
            param.requires_grad = False

    model = LitWrapper(Recformer, learning_rate=args.learning_rate)
    # integrates Recformer with PyTorch Lightning, a library that simplifies the training of PyTorch models.
    # model wrapper: to choose model, and set hyperparameters

    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="avg_val_accuracy", mode="max", filename="{epoch}-{avg_val_accuracy:.4f}")
    # a feature of PyTorch Lightning that automatically saves the model's state. 
    # save the top 5 models based on average validation 
    # avg_val_accuracy: after each epoch, the average validation accuracy is calculated


    # setting PyTorch Lightning trainer
    trainer = Trainer(accelerator="gpu",
                     max_epochs=args.num_train_epochs,
                     devices=args.device,
                     accumulate_grad_batches=args.gradient_accumulation_steps, # cy: what does this arguments mean?
                     val_check_interval=args.valid_step,# [?] doing evalution every echo? OR every 2000 steps?
                     default_root_dir=args.output_dir,
                     gradient_clip_val=1.0,             # avoid gradient exploding, clip the gradient norm to 1.0 
                     log_every_n_steps=args.log_step,   # how often to log within steps
                     precision=16 if args.fp16 else 32, # the numerical precision of the weights
                     strategy='deepspeed_stage_2',      # the distributed training strategy
                     callbacks=[checkpoint_callback]    # call back at certain points
                     )

    # start training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)



if __name__ == "__main__":
    main()