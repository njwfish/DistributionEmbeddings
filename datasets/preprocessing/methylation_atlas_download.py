import os
import GEOparse
import ftplib
from urllib.parse import urlparse
import wget


# Replace with your GEO Series accession
GSE_ID = "GSE186458"
OUTPUT_DIR = "/orcd/data/omarabu/001/gokul/DistributionEmbeddings/data/geo_downloads"

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load GEO Series metadata
print(f"Fetching metadata for {GSE_ID}...")
gse = GEOparse.get_GEO(geo=GSE_ID, destdir="/orcd/data/omarabu/001/gokul/DistributionEmbeddings/data/geo_downloads")

# Iterate through samples
for gsm_id, gsm in gse.gsms.items():
    print(f"\nProcessing {gsm_id}...")

    for supp_file in [x[1][0] for x in gsm.metadata.items() if 'supplementary_file' in x[0]]:
        if supp_file.endswith("hg38.pat.gz"):
            filename = f"{gsm_id}_{os.path.basename(supp_file)}"
            out_path = os.path.join(OUTPUT_DIR, filename)

            if not os.path.exists(out_path):
                print(f"Downloading {supp_file} → {filename}")
                wget.download(supp_file, out_path)
                break  # remove this line if more than one .pat.gz per sample
        else:
            print(f"Skipping non-pat.gz: {supp_file}")