from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from typing import Optional
import gzip
import os
import logging

logger = logging.getLogger(__name__)

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
        
        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'

        self.tokenized_data_file = f'{self.data_dir}/pfam_tokenized_data.pt'

        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._tokenize_data(lines_to_read=lines_to_read)
        self.data = torch.load(self.tokenized_data_file)
    
    def _tokenize_data(self, lines_to_read=10**6):

        f = gzip.open(self.data_dir+'/Pfam-A.fasta.gz', 'rt')
        d = {}
        i = 0

        logger.info('building pfam dict')

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

        logger.info('tokenizing pfam data')

        for fam, seqs in d.items():
            if len(seqs) < self.set_size:
                continue

            # how many sets can we make? 
            n_sets = min(len(seqs) // self.set_size, self.max_sets_per_fam)
            np.random.shuffle(seqs)

            for i in range(n_sets):
                batch = seqs[i*self.set_size : (i+1)*self.set_size]

                esm_input_ids = []
                esm_attention_mask = []
                progen_input_ids = []
                progen_attention_mask = []
                texts = []

                for seq in batch:

                    pg2 = self._tokenize_for_progen(seq)
                    progen_input_ids.append(pg2[0])
                    progen_attention_mask.append(pg2[1])
                    esm = self._tokenize_for_esm(seq)
                    esm_input_ids.append(esm[0])
                    esm_attention_mask.append(esm[1])
                    texts.append(seq[:self.max_length]) # note truncated


                esm_input_ids = torch.stack(esm_input_ids)
                esm_attention_mask = torch.stack(esm_attention_mask)
                progen_input_ids = torch.stack(progen_input_ids)
                progen_attention_mask = torch.stack(progen_attention_mask)

                tokenized_data.append({
                    'samples' : {
                    'esm_input_ids': esm_input_ids,
                    'esm_attention_mask': esm_attention_mask,
                    'progen_input_ids': progen_input_ids,
                    'progen_attention_mask': progen_attention_mask,},
                    'pfam': fam,
                    'raw_texts': texts
                })

        torch.save(tokenized_data, self.tokenized_data_file)
        logger.info(f"Tokenized data saved to {self.tokenized_data_file}")
        f.close()

    def _tokenize_for_esm(self, sequence):
        """
        Tokenize a protein sequence for ESM.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # ESM tokenizer requires starting with <cls> token
        # Ensure the sequence is not modified with extra spaces or newlines
        sequence = sequence.strip()
        
        # Tokenize with appropriate settings and explicitly add special tokens
        tokens = self.esm_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True,  # This will add the CLS token
            return_tensors='pt'
        )
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def _tokenize_for_progen(self, sequence):
        """
        Tokenize a protein sequence for Progen.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Clean the sequence
        sequence = sequence.strip()
        
        # Since the tokenizer isn't automatically adding special tokens,
        # we'll manually add BOS and EOS tokens
        bos_token = self.progen_tokenizer.bos_token
        eos_token = self.progen_tokenizer.eos_token
        
        # Ensure sequence starts with BOS and ends with EOS
        if bos_token and not sequence.startswith(bos_token):
            sequence = bos_token + sequence
        
        if eos_token and not sequence.endswith(eos_token):
            sequence = sequence + eos_token
        
        # Tokenize with appropriate settings
        # Set add_special_tokens=False since we've manually added them
        tokens = self.progen_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=False,  # Don't add again since we did it manually
            return_tensors='pt'
        )
        
        # Log the first sequence's token IDs for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Progen tokenized sequence: {tokens.input_ids[0]}")
            logger.debug(f"Progen BOS token ID: {self.progen_tokenizer.convert_tokens_to_ids(bos_token)}")
            logger.debug(f"Progen EOS token ID: {self.progen_tokenizer.convert_tokens_to_ids(eos_token)}")
        
        return tokens.input_ids[0], tokens.attention_mask[0]

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
            'raw_texts': tuple(item['raw_texts'])
        }
