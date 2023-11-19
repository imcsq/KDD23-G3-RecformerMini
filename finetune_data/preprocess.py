import os, json, gzip, requests, argparse
from tqdm import tqdm
from clint.textui import progress
from collections import defaultdict

def download(url, export_path):
    r = requests.get(url, stream=True)
    with open(export_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()

def download_data(categories, folderOfCat):
    for cat in tqdm(categories, desc='Download datasets'):
        folder=folderOfCat[cat]
        os.makedirs(folder,exist_ok=True)
        meta_link=f'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_{cat}.json.gz'
        download(meta_link,os.path.join(folder,meta_link.split('/')[-1]))
        review_link=f'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/{cat}_5.json.gz'
        download(review_link,os.path.join(folder,review_link.split('/')[-1]))

def get_meta_data(meta_fp):
    meta_data=dict()
    with gzip.open(meta_fp) as f:
        for l in tqdm(f, desc=f'Add item IDs from {meta_fp} to meta data'):
            obj=json.loads(l); asin=obj['asin']
            if ('title' in obj) and ('brand' in obj) and ('category' in obj): 
                meta_data[asin]={
                'title': obj['title'], 
                'brand': obj['brand'], 
                'category': ' '.join(obj['category'])
                }
    return meta_data

def prepare_meta_data(categories, folderOfCat):
    for cat in categories:
        folder=folderOfCat[cat]
        filename=f'meta_{cat}.json.gz'
        meta_data=get_meta_data(os.path.join(folder,filename))
        out_path=os.path.join(folder,'meta_data.json')
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(meta_data,f)

def load_meta_data(path):
    # Read meta data of all valid items
    assert os.path.exists(path), f"{path} does not exist!"
    with open(path, 'r') as f: meta_data=json.load(f); return meta_data

def get_review_seqs(review_fp, meta_data):
    seq_dict=defaultdict(list)
    with gzip.open(review_fp) as f:
        filename=review_fp.split('/')[-1]
        for l in tqdm(f, f'Prepare review sequences of {filename}'):
            obj=json.loads(l)
            if obj['asin'] in meta_data:
                seq_dict[obj["reviewerID"]].append((obj['unixReviewTime'], obj['asin']))
    return seq_dict

def remap_seqs(seq_dict):
    def remap_id(oldID, map): 
        if oldID not in map: map[oldID]=len(map)
        return map[oldID]
    reviewerIDMap,itemIDMap=dict(),dict()
    remapped_seq_dict=defaultdict(list)
    for reviewerID, seq in tqdm(seq_dict.items(), desc=f'Remap user and item IDs'):
        if len(seq)>3:
            new_reviewerID=remap_id(reviewerID, reviewerIDMap)
            remapped_seq_dict[new_reviewerID]=[(reviewTime,remap_id(itemID,itemIDMap)) for reviewTime,itemID in seq]
    return reviewerIDMap, itemIDMap, remapped_seq_dict

def sort_seqs(seq_dict):
    for reviewerID, seq in tqdm(seq_dict.items(), desc='Sort sequences by time'):
        seq.sort(key=lambda x:x[0])
        seq_dict[reviewerID]=[e[1] for e in seq]

def split_seq(seq_dict):
    train_seqs, val_seqs, test_seqs=dict(), dict(), dict()
    for reviewerID, seq in tqdm(seq_dict.items(), desc='Split each sequence into training, validation, testing data'):
        if len(seq)<3: train_seqs[reviewerID]=seq
        else: train_seqs[reviewerID], val_seqs[reviewerID], test_seqs[reviewerID]=seq[:-2], seq[-2:-1], seq[-1:]
    return train_seqs, val_seqs, test_seqs

def save_data(data_dict):
    for export_path, data in data_dict.items():
        print(f'Export {export_path}')
        with open(export_path,'w',encoding='utf8') as f:
            json.dump(data,f)

def process_review_sequences(categories, folderOfCat):
    for cat in tqdm(categories, desc='Process review sequences'):
        folder=folderOfCat[cat]
        review_path=os.path.join(folder,f'{cat}_5.json.gz')
        meta_data=load_meta_data(os.path.join(folder,'meta_data.json'))
        seq_dict=get_review_seqs(review_path,meta_data)
        reviewerIDMap, itemIDMap, remapped_seq_dict=remap_seqs(seq_dict)
        sort_seqs(remapped_seq_dict)
        train_seqs, val_seqs, test_seqs=split_seq(remapped_seq_dict)
        save_data({
            os.path.join(folder,'umap.json'): reviewerIDMap,
            os.path.join(folder,'smap.json'): itemIDMap,
            os.path.join(folder,'train.json'): train_seqs,
            os.path.join(folder,'val.json'): val_seqs,
            os.path.join(folder,'test.json'): test_seqs
        })
    print('Finished!')
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='Data preprocessing',
                    description='Download and preprocess meta data and review data')
    parser.add_argument('-d', '--download', action='store_true') # download the dataset or not
    parser.add_argument('-m', '--metadata', action='store_true') # extract metadata or not
    parser.add_argument('-s', '--sequence', action='store_true') # extract sequence or not
    args=parser.parse_args()
    
    # Six categories for evaluation
    categories = ['Industrial_and_Scientific', 'Musical_Instruments', 'Arts_Crafts_and_Sewing', 'Office_Products', 'Video_Games', 'Pet_Supplies']
    folders = ['Scientific', 'Instruments', 'Arts', 'Office', 'Games', 'Pet']
    folderOfCat = {c:f for c,f in zip(categories, folders)}
    
    if(args.download): download_data(categories, folderOfCat)
    
    if(args.metadata): prepare_meta_data(categories,folderOfCat)
    
    if(args.sequence): process_review_sequences(categories, folderOfCat)