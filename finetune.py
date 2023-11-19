import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from pytorch_lightning import seed_everything

#from utils import read_json, AverageMeterSet, Ranker
#from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset

from torch.optim import AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that increases linearly for a warmup period
    and then decreases linearly after.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num_decay_steps = max(1, num_training_steps - num_warmup_steps)
        return max(
            0.0, 1 - float(current_step - num_warmup_steps) / float(num_decay_steps)
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_optimizer_and_scheduler(model: nn.Module, num_train_optimization_steps, args):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)
    return optimizer, scheduler


import json
import torch
import torch.nn as nn

MAX_VAL = 1e4

def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        
        data = {}
        for key, value in raw.items():
            if as_int:
                data[int(key)] = value
            else:
                data[key] = value
        
        del raw
        return data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class Ranker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, scores, labels):
        labels = labels.squeeze()
        
        try:
            loss = self.ce(scores, labels).item()
        except:
            print(scores.size())
            print(labels.size())
            loss = 0.0
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC

        return res + [loss]








def load_data(args):

    # Read train data
    train_path = os.path.join(args.data_path, args.train_file)
    train = read_json(train_path, True)

    # Read validation data
    val_path = os.path.join(args.data_path, args.dev_file)
    val = read_json(val_path, True)

    # Read test data
    test_path = os.path.join(args.data_path, args.test_file)
    test = read_json(test_path, True)

    # Read item metadata
    meta_path = os.path.join(args.data_path, args.meta_file)
    item_meta_dict = json.load(open(meta_path))

    # Read item to id mapping
    item2id_path = os.path.join(args.data_path, args.item2id_file)
    item2id = read_json(item2id_path)

    # Create id to item mapping
    id2item = {v: k for k, v in item2id.items()}

    # Filter item metadata based on item2id
    item_meta_dict_filtered = {k: v for k, v in item_meta_dict.items() if k in item2id}

    return train, val, test, item_meta_dict_filtered, item2id, id2item


tokenizer_glb: RecformerTokenizer = None
def par_tokenize_doc(doc, tokenizer_glb):

    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):
    model.eval()

    items = [tokenized_items[key] for key in sorted(tokenized_items)]

    item_embeddings = []

    with torch.no_grad():
        batch_start = 0
        while batch_start < len(items):
            batch_end = min(batch_start + args.batch_size, len(items))

            item_batch = [[item] for item in items[batch_start:batch_end]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for key, value in inputs.items():
                inputs[key] = torch.LongTensor(value).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

            batch_start += args.batch_size

    item_embeddings = torch.cat(item_embeddings, dim=0)

    return item_embeddings


def eval(model, dataloader, args):

    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    with torch.no_grad():
        for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):

            for k, v in batch.items():
                batch[k] = v.to(args.device)
            labels = labels.to(args.device)

            scores = model(**batch)

            res = ranker(scores, labels)

            metrics = {}
            for i, k in enumerate(args.metric_ks):
                metrics["NDCG@%d" % k] = res[2*i]
                metrics["Recall@%d" % k] = res[2*i+1]
            metrics["MRR"] = res[-3]
            metrics["AUC"] = res[-2]

            for k, v in metrics.items():
                average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    return average_metrics

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        with autocast(enabled=args.fp16):
            loss = model(**batch)
        
        loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scheduler.step()

    if not args.fp16:
        scheduler.step()  # Update learning rate schedule
        optimizer.step()
        optimizer.zero_grad()

def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)
    

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode='val', collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)

    
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            collate_fn=val_data.collate_fn)
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    else:
        print(f'Encoding items.')
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)
    
    item_embeddings = torch.load(path_item_embeddings)
    model.init_item_embedding(item_embeddings)

    model.to(args.device) # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    test_metrics = eval(model, test_loader, args)
    print(f'S1TestRes@{path_output}: {test_metrics}')
    with open("finetune_res.txt", 'a') as file:
        file.write(f'S1TestRes@{path_output}: {test_metrics}\n')    
    best_target = float('-inf')
    patient = 5

    for epoch in range(args.num_train_epochs):

        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                torch.save(model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break
    
    print('Load best model in stage 1.')
    model.load_state_dict(torch.load(path_ckpt))

    patient = 3

    for epoch in range(args.num_train_epochs):

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 3
                torch.save(model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break

    print('Test with the best checkpoint.')  
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = eval(model, test_loader, args)
    print(f'S2TestRes@{path_output}: {test_metrics}')
    with open("finetune_res.txt", 'a') as file:
        file.write(f'S2TestRes@{path_output}: {test_metrics}\n')
               
if __name__ == "__main__":
    main()
