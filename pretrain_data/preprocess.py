import os, json, gzip, requests, random, time, argparse
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

def download_data(categories):
    links= {
        'meta': [f'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_{c}.json.gz' for c in categories],
        'review': [f'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/{c}_5.json.gz' for c in categories]
    }
    for form, lst in links.items():
        if not os.path.exists(form): os.mkdir(form)
        random.seed(time.time()); random.shuffle(lst)
        for path in tqdm(lst, desc='Download datasets'):
            filename=path.split('/')[-1]
            download(path, f'./{form}/{filename}')
            
def add_asins(gzip_fp, filter, asins):
    with gzip.open(gzip_fp) as f:
        for l in tqdm(f,desc=f'Extract item IDs from {gzip_fp}'):
            obj=json.loads(l)
            if (filter in obj) and obj[filter]!=None and obj['asin']!=None: 
                asins.add(obj['asin'])

def add_meta_data(meta_fp, valid_asins, meta_data):
    with gzip.open(meta_fp) as f:
        for l in tqdm(f, desc=f'Add item IDs from {meta_fp} to meta data'):
            obj=json.loads(l); asin=obj['asin']
            if asin in valid_asins and ('title' in obj) and ('brand' in obj) and ('category' in obj): 
                meta_data[asin]={
                'title': obj['title'], 
                'brand': obj['brand'], 
                'category': ' '.join(obj['category'])
                }

def add_review_seqs(review_fp, seq_dict, meta_data):
    with gzip.open(review_fp) as f:
        filename=review_fp.split('/')[-1]
        for l in tqdm(f, f'Add review sequences from {filename}'):
            obj=json.loads(l)
            if obj['asin'] in meta_data:
                seq_dict[f'{filename}-{obj["reviewerID"]}'].append((obj['unixReviewTime'], obj['asin']))

def sort_review_seqs(seq_dict):
    for seq_id, seq in tqdm(seq_dict.items(), desc="Sort review sequences by time"):
        seq.sort()
        seq_dict[seq_id]=[e[1] for e in seq]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='Data preprocessing',
                    description='Download and preprocess meta data and review data')
    parser.add_argument('-d', '--download', action='store_true') # download the dataset or not
    parser.add_argument('-m', '--metadata', action='store_true') # extract metadata or not
    parser.add_argument('-s', '--sequence', action='store_true') # extract sequence or not
    args=parser.parse_args()
    
    # First six categories are for pre-training, the last category is for validation
    categories = ['Automotive', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', 'Movies_and_TV', 'CDs_and_Vinyl']

    if(args.download): download_data(categories)
    
    if(args.metadata):
        # Amazon Standard Identification Numbers in meta data and review data
        asins_dict={'meta':set(),'review':set()}
        
        # Obtain the meta data and review data paths
        path_dict = {category: {'meta': os.path.join('meta', 'meta_'+category+'.json.gz'), 
                                'review': os.path.join('review', category+'_5.json.gz')} 
                    for category in categories}

        # Assuming all file paths exist
        assert all([os.path.exists(path) for cat_paths in path_dict.values() for path in cat_paths.values()]), "Data is not fully downloaded!"

        # Obtain the ASIN set of meta data and review data respectively
        for category, cat_paths in path_dict.items():
            for form, path in cat_paths.items():
                add_asins(path, 'title' if form=='meta' else 'reviewerID', asins_dict[form])

        # Obtain common ASINs exist in both meta data and review data
        asins_dict['intersection']=asins_dict['meta'] and asins_dict['review']

        # Obtain the meta data of common ASINs
        meta_data = dict()
        for path in tqdm([cat_paths['meta'] for cat_paths in path_dict.values()], desc='Obtain meta data'): 
            add_meta_data(path, asins_dict['intersection'], meta_data)

        # Export the meta data
        with open('meta_data.json', 'w', encoding='utf8') as f:
            json.dump(meta_data, f)
    
    if args.sequence:
        # File paths of review data
        review_paths=[os.path.join('review', category+'_5.json.gz') for category in categories]
        
        # Assuming all file paths exist
        assert all([os.path.exists(path) for path in review_paths]), "Review data is not fully downloaded!"
        
        # Read meta data of all valid items
        assert os.path.exists('meta_data.json'), "meta data is not prepared yet!"
        with open('meta_data.json', 'r') as f: meta_data=json.load(f)
        
        # Extract review sequences by category and user id
        pretrain_seqs, validate_seqs=defaultdict(list), defaultdict(list)
        for path in review_paths:
            if path==review_paths[-1]: add_review_seqs(path,validate_seqs,meta_data)
            else: add_review_seqs(path,pretrain_seqs,meta_data)
        sort_review_seqs(pretrain_seqs); sort_review_seqs(validate_seqs)
        
        # Export the pretrain data and validation data
        print('Export train.json')
        with open('train.json', 'w') as f:
            pretrain_data=list(pretrain_seqs.values()); json.dump(pretrain_data, f)
        print('Export dev.json')
        with open('dev.json', 'w') as f:
            val_data=list(validate_seqs.values()); json.dump(val_data, f)
        print('Finished!')