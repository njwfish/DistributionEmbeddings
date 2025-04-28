# process_one.py
import gzip, os, sys, subprocess, random, tempfile, csv
from pyfaidx import Fasta

tool_path = '/orcd/data/omarabu/001/gokul/wgbs_tools'
genome = Fasta('/orcd/data/omarabu/001/gokul/wgbs_tools/references/hg38/hg38.fa', as_raw=True)
os.environ["PATH"] = f"{tool_path}:{os.environ['PATH']}"

window = 100
n_to_sample = 10**6

fpath = sys.argv[1]
out_csv = fpath.replace('.hg38.pat.gz', '_seqs.csv')

# if os.path.exists(out_csv):
#     sys.exit(0)  # already done!

with gzip.open(fpath, 'rb') as f:
    lines = f.read().decode().splitlines()

sample = random.sample(lines, min(len(lines), n_to_sample))
sid_cpgs = []
cpg_map = {}

for line in sample:
    try:
        parts = line.split()
        sid = int(parts[1])
        cpg = parts[2].lower()
        if '.' in cpg:
            continue
        sid_cpgs.append(sid)
        cpg_map[sid] = cpg
    except:
        continue

if not sid_cpgs:
    sys.exit(0)  # no good data :(

with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
    site_file = tmp.name
    for sid in sid_cpgs:
        tmp.write(f"{sid}\n")

res = subprocess.run([
    'wgbstools', 'convert', '--site_file', site_file,
    '--genome', 'hg38',
], capture_output=True, text=True)

os.remove(site_file)

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

        cpg = cpg_map.get(sid).upper()
        if not cpg:
            continue
        gseq = genome[chr][start-window:start].lower()
        writer.writerow([f"{gseq}{cpg}", f"{chr}:{start}"])
