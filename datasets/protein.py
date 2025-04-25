from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from typing import Optional
import gzip
import os


class PfamDataset(Dataset):
    def __init__(self,
                 data_dir: str = 'data/pfam',
                 set_size: int = 16,
                 esm_name: str = 'facebook/esm2_t6_8M_UR50D',
                 progen_name: str = 'hugohrban/progen2-medium',
                 max_length: int = 512,
                 seed: Optional[int] = 212121,
                 tokenize: bool = False,
                 lines_to_read: int = 10**6,
                 max_sets_per_fam: int = 100):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = data_dir
        self.set_size = set_size
        self.max_length = max_length
        self.max_sets_per_fam = max_sets_per_fam
        
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_name, trust_remote_code=True)
        self.progen_tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)
        
        self.progen_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

        self.tokenized_data_file = f'{self.data_dir}/pfam_tokenized_data.pt'

        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._tokenize_data(lines_to_read=lines_to_read)
        self.data = torch.load(self.tokenized_data_file)
    
    def _tokenize_data(self, lines_to_read=10**6):

        f = gzip.open(self.data_dir+'/Pfam-A.fasta.gz', 'rt')
        d = {}
        i = 0

        # collect seqs per pfam
        for line in f:
            if line.startswith('>'):
                fam = line.split()[-1].split(';')[0]
                d.setdefault(fam, [])
            else:
                d[fam].append(line.strip())
            i += 1
            if i > lines_to_read:
                break

        tokenized_data = []

        for fam, seqs in d.items():
            if len(seqs) < self.set_size:
                continue

            # how many sets can we make? 
            n_sets = min(len(seqs) // self.set_size, self.max_sets_per_fam)
            np.random.shuffle(seqs)

            for i in range(n_sets):
                batch = seqs[i*self.set_size : (i+1)*self.set_size]

                pg2 = self.progen_tokenizer(batch,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                esm = self.esm_tokenizer(batch,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                tokenized_data.append({
                    'samples' : {
                    'esm_input_ids': esm['input_ids'],
                    'esm_attention_mask': esm['attention_mask'],
                    'progen_input_ids': pg2['input_ids'],
                    'progen_attention_mask': pg2['attention_mask'],},
                    'pfam': fam,
                    'raw_texts': batch
                })

        torch.save(tokenized_data, self.tokenized_data_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        esm_input_ids = item['samples']['esm_input_ids']
        esm_attention_mask = item['samples']['esm_attention_mask']
        progen_input_ids = item['samples']['progen_input_ids']
        progen_attention_mask = item['samples']['progen_attention_mask']

        return { 'samples' : {
            'esm_input_ids': esm_input_ids,
            'esm_attention_mask': esm_attention_mask,
            'progen_input_ids': progen_input_ids,
            'progen_attention_mask': progen_attention_mask},
            'pfam': item['pfam'],
            'raw_texts': item['raw_texts']
        }
