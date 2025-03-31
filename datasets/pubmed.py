import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
from transformers import BertTokenizer, GPT2Tokenizer
import random
import requests
import gzip
import shutil
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import logging
import time
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PubMedDataset(Dataset):
    """Dataset for PubMed abstracts organized by MeSH major topics."""
    
    def __init__(
        self,
        data_dir: str = "data/pubmed",
        set_size: int = 20,   # Number of documents per set
        min_docs_per_tag: int = 50,    # Minimum docs required for a tag to be included
        bert_model_name: str = "bert-base-uncased",
        gpt2_model_name: str = "gpt2",
        max_bert_length: int = 128,    # Max token length for BERT
        max_gpt2_length: int = 256,    # Max token length for GPT-2
        download: bool = True,
        force_reprocess: bool = False,  # Force reprocessing of data
        num_files_to_download: int = 5,  # Number of PubMed XML files to download (1-1274)
        start_file_num: int = 1,  # Starting file number
        seed: Optional[int] = 42,
    ):
        """
        Initialize the PubMed dataset.
        
        Args:
            data_dir: Directory to store the PubMed data
            set_size: Number of documents per set
            min_docs_per_tag: Minimum docs required for a tag to be included
            bert_model_name: BERT model name for tokenization
            gpt2_model_name: GPT-2 model name for tokenization
            max_bert_length: Maximum token length for BERT
            max_gpt2_length: Maximum token length for GPT-2
            download: Whether to download the dataset if not present
            force_reprocess: Force reprocessing of data even if cached files exist
            num_files_to_download: Number of PubMed XML files to download (1-1274)
            start_file_num: Starting file number for downloading
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.data_dir = data_dir
        self.set_size = set_size
        self.min_docs_per_tag = min_docs_per_tag
        self.max_bert_length = max_bert_length
        self.max_gpt2_length = max_gpt2_length
        self.num_files_to_download = min(num_files_to_download, 1274)  # Maximum 1274 files
        self.start_file_num = max(1, start_file_num)  # Minimum file number is 1
        
        # Create directories
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.processed_data_dir = os.path.join(data_dir, "processed")
        self.tokenized_data_dir = os.path.join(data_dir, "tokenized")
        
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.tokenized_data_dir, exist_ok=True)
        
        # Initialize tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        # File paths for processed data
        self.processed_articles_file = os.path.join(self.processed_data_dir, "pubmed_articles.json")
        self.processed_sets_file = os.path.join(self.processed_data_dir, "pubmed_sets.json")
        self.tokenized_data_file = os.path.join(self.tokenized_data_dir, 
                                               f"pubmed_tokenized_bert{max_bert_length}_gpt2{max_gpt2_length}.pt")
        
        # Check if we need to download or process data
        if download and (force_reprocess or not os.path.exists(self.tokenized_data_file)):
            # We need to get the raw data first
            if force_reprocess or not os.path.exists(self.processed_sets_file):
                # We need to download and/or process the raw data
                self._download_and_process_pubmed(force_reprocess)
            
            # Now tokenize the processed data
            self._tokenize_data()
        
        # Load the tokenized data
        if os.path.exists(self.tokenized_data_file):
            self._load_tokenized_data()
        else:
            raise FileNotFoundError(
                f"Tokenized data file {self.tokenized_data_file} not found. "
                f"Set download=True to download and process the dataset."
            )
    
    def _download_and_process_pubmed(self, force_reprocess=False):
        """Download and process the PubMed dataset."""
        logger.info("Downloading and processing PubMed dataset...")
        
        # Step 1: Download PubMed XML files if needed
        self._download_pubmed_files()
        
        # Step 2: Parse the XML files and extract articles
        if force_reprocess or not os.path.exists(self.processed_articles_file):
            self._extract_articles_from_xml()
        
        # Step 3: Create document sets based on MeSH terms
        if force_reprocess or not os.path.exists(self.processed_sets_file):
            self._create_document_sets()
    
    def _download_pubmed_files(self):
        """Download PubMed baseline XML files."""
        # Create directory for compressed files
        gz_dir = os.path.join(self.raw_data_dir, "gz")
        xml_dir = os.path.join(self.raw_data_dir, "xml")
        os.makedirs(gz_dir, exist_ok=True)
        os.makedirs(xml_dir, exist_ok=True)
        
        # Calculate which files to download
        end_file_num = min(self.start_file_num + self.num_files_to_download - 1, 1274)
        file_nums = list(range(self.start_file_num, end_file_num + 1))
        
        logger.info(f"Downloading PubMed files {self.start_file_num} to {end_file_num}...")
        
        for file_num in tqdm(file_nums, desc="Downloading PubMed files"):
            # Format: pubmed22n0001.xml.gz where 22 is the baseline year and 0001 is the file number
            file_name = f"pubmed25n{file_num:04d}.xml.gz"
            baseline_url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{file_name}"
            gz_file = os.path.join(gz_dir, file_name)
            xml_file = os.path.join(xml_dir, file_name.replace('.gz', ''))
            
            # Skip if XML file already exists
            if os.path.exists(xml_file):
                logger.info(f"File {xml_file} already exists, skipping download")
                continue
            
            # Download the file if it doesn't exist
            if not os.path.exists(gz_file):
                logger.info(f"Downloading {baseline_url}")
                response = requests.get(baseline_url, stream=True)
                response.raise_for_status()  # Raise an error for bad responses
                
                print("Downloading file", gz_file)
                with open(gz_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Extract the file
            if not os.path.exists(xml_file):
                logger.info(f"Extracting {gz_file}")
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(xml_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    
    def _extract_articles_from_xml(self):
        """Extract articles from the downloaded XML files."""
        logger.info("Extracting articles from XML files...")
        
        xml_dir = os.path.join(self.raw_data_dir, "xml")
        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
        # sort the xml files by the file number
        xml_files.sort()
        
        if not xml_files:
            logger.warning("No XML files found. Using synthetic data instead.")
            self._create_synthetic_data()
            return
        
        # Process each XML file and extract articles
        all_articles = []
        
        for xml_file in tqdm(xml_files, desc="Processing XML files"):
            logger.info(f"Processing {xml_file}")
            articles = self._parse_xml_file(xml_file)
            all_articles.extend(articles)
            logger.info(f"Extracted {len(articles)} articles from {xml_file}")

        logger.info(f"Total articles extracted: {len(all_articles)}")
        
        # Save the extracted articles
        with open(self.processed_articles_file, 'w') as f:
            json.dump(all_articles, f)
        
        logger.info(f"Saved extracted articles to {self.processed_articles_file}")
    
    def _parse_xml_file(self, xml_file):
        """Parse a single XML file and extract articles."""
        articles = []

        # Use iterparse to avoid loading the entire file into memory
        context = ET.iterparse(xml_file, events=('end',))
        
        current_article = None
        root = None  # Keep a reference to the root
        
        for event, elem in context:
            # Get reference to the root
            if root is None:
                root = elem.getroot() if hasattr(elem, 'getroot') else elem
            
            if elem.tag == 'PubmedArticle':
                # Extract PMID
                pmid_elem = elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else None
                
                # Extract title
                title_elem = elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Extract abstract
                abstract_text = ""
                abstract_elements = elem.findall(".//AbstractText")
                for abstract_elem in abstract_elements:
                    if abstract_elem.text:
                        abstract_text += abstract_elem.text + " "
                
                # Extract MeSH terms, focusing on major topics
                mesh_terms = []
                descriptor_elements = elem.findall(".//MeshHeading/DescriptorName")
                for desc_elem in descriptor_elements:
                    if desc_elem.get("MajorTopicYN") == "Y" and desc_elem.text:
                        mesh_terms.append(desc_elem.text)
                
                if pmid and title and abstract_text and mesh_terms:
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract_text.strip(),
                        "mesh_terms": mesh_terms
                    })

                # Clear the element to save memory
                elem.clear()
        
        # Clear the root element after processing
        if root is not None:
            root.clear()
        
        return articles
    
    def _create_document_sets(self):
        """Create document sets based on MeSH terms."""
        logger.info("Creating document sets from articles...")
        
        # Load the processed articles
        with open(self.processed_articles_file, 'r') as f:
            articles = json.load(f)
        
        # Group articles by MeSH terms
        mesh_to_articles = defaultdict(list)
        for article in articles:
            for term in article["mesh_terms"]:
                mesh_to_articles[term].append(article)
        
        # Select MeSH terms with sufficient articles
        valid_mesh_terms = [
            term for term, term_articles in mesh_to_articles.items()
            if len(term_articles) >= self.min_docs_per_tag
        ]
        
        logger.info(f"Selected {len(valid_mesh_terms)} valid MeSH terms")
        
        # Create the final dataset structure
        processed_data = []
        for term in valid_mesh_terms:
            term_articles = mesh_to_articles[term]
        
            
            # Ensure we have at least set_size documents
            if len(term_articles) < self.set_size:
                logger.warning(f"Term {term} has fewer than {self.set_size} articles, skipping")
                continue
                
            # Create sets of documents
            num_sets = len(term_articles) // self.set_size
            for i in range(num_sets):
                set_articles = term_articles[i*self.set_size:(i+1)*self.set_size]
                
                # Extract text from articles
                texts = [f"{article['title']} {article['abstract']}" for article in set_articles]
                
                processed_data.append({
                    "mesh_term": term,
                    "texts": texts,
                    "pmids": [article['pmid'] for article in set_articles]
                })
        
        # Save the processed sets
        with open(self.processed_sets_file, 'w') as f:
            json.dump(processed_data, f)
        
        logger.info(f"Created {len(processed_data)} document sets")
    
    def _tokenize_data(self):
        """Tokenize the processed data for BERT and GPT-2."""
        logger.info("Tokenizing data for BERT and GPT-2...")
        
        # Load the processed sets
        with open(self.processed_sets_file, 'r') as f:
            raw_data = json.load(f)
        
        # Process data into a format suitable for training
        tokenized_data = []
        
        for item in tqdm(raw_data, desc="Tokenizing documents"):
            mesh_term = item["mesh_term"]
            texts = item["texts"]
            pmids = item.get("pmids", [])
            
            # Tokenize texts for BERT
            bert_encodings = self.bert_tokenizer(
                texts, 
                padding='max_length',
                truncation=True,
                max_length=self.max_bert_length,
                return_tensors='pt'
            )
            
            # Tokenize texts for GPT-2
            gpt2_encodings = self.gpt2_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_gpt2_length,
                return_tensors='pt'
            )
            
            tokenized_data.append({
                "mesh_term": mesh_term,
                "bert_input_ids": bert_encodings["input_ids"],
                "bert_attention_mask": bert_encodings["attention_mask"],
                "gpt2_input_ids": gpt2_encodings["input_ids"],
                "gpt2_attention_mask": gpt2_encodings["attention_mask"],
                "raw_texts": texts,
                "pmids": pmids
            })

        logger.info(f"Tokenized {len(tokenized_data)} document sets")
        
        # Save the tokenized data
        torch.save(tokenized_data, self.tokenized_data_file)
        
        logger.info(f"Saved tokenized data to {self.tokenized_data_file}")
    
    def _load_tokenized_data(self):
        """Load the tokenized data."""
        logger.info(f"Loading tokenized data from {self.tokenized_data_file}")
        
        self.data = torch.load(self.tokenized_data_file)
        
        logger.info(f"Loaded {len(self.data)} document sets")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
                
        return {
            'mesh_term': item["mesh_term"],
            'samples': {
                'bert_input_ids': item["bert_input_ids"],
                'bert_attention_mask': item["bert_attention_mask"],
                'gpt2_input_ids': item["gpt2_input_ids"],
                'gpt2_attention_mask': item["gpt2_attention_mask"]
            },
            'raw_texts': item["raw_texts"],
            'pmids': item.get("pmids", [])
        } 