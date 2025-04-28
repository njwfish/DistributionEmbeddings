import gzip
import os
import subprocess
import random
import tempfile
import csv
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pyfaidx import Fasta

# setup :)
tool_path = '/orcd/data/omarabu/001/gokul/wgbs_tools'
data_dir = '/orcd/data/omarabu/001/gokul/DistributionEmbeddings/data/geo_downloads/'
genome = Fasta('/orcd/data/omarabu/001/gokul/wgbs_tools/references/hg38/hg38.fa', as_raw=True)
os.environ["PATH"] = f"{tool_path}:{os.environ['PATH']}"

window = 100
n_to_sample = 10**5

# worker function!
def process_file(d):
    if not d.endswith('.gz'):
        return

    fpath = os.path.join(data_dir, d)
    out_csv = fpath.replace('.hg38.pat.gz', '_seqs.csv')
    # if os.path.exists(out_csv):
    #     return  # done already :)

    with gzip.open(fpath, 'rb') as f:
        lines = f.read().decode().splitlines()

    if len(lines) < n_to_sample:
        sample = lines
    else:
        sample = random.sample(lines, n_to_sample)

    sid_cpgs = []
    cpg_map = {}

    for line in sample:
        parts = line.split()
        sid = int(parts[1])
        cpg = parts[2].lower()
        if '.' in cpg:
            continue
        sid_cpgs.append(sid)
        cpg_map[sid] = cpg

    if not sid_cpgs:
        return  # skip empty!

    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        site_file = tmp.name
        for sid in sid_cpgs:
            tmp.write(f"{sid}\n")

    res = subprocess.run([
        'wgbstools', 'convert', '--site_file', site_file,
        '--genome', 'hg38',
    ], capture_output=True, text=True)

    os.remove(site_file)  # clean temp :) 

    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['seq', 'loc'])

        for line in res.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.strip().split()
            chr = parts[0]
            start = int(parts[1])
            sid = int(parts[3])

            cpg = cpg_map.get(sid, '').upper()
            if not cpg:
                continue

            gseq = genome[chr][start-window:start].lower()
            writer.writerow([f"{gseq}{cpg}", f"{chr}:{start}"])

# parallel magic!
if __name__ == "__main__":
    files = os.listdir(data_dir)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, files), total=len(files)))
