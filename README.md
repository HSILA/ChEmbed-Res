# MTEB Evaluation Scripts

## Setup
To set up the environment, run the installation script:
```bash
./install.sh
source .venv/bin/activate
```

## Usage
```bash
python chemrxiv_bench.py   # 53 models × ChemRxivRetrieval
python nomic_bench.py      # 6 models × MTEB(eng,v2) + ChemTEB
```

## Output
- `results/chemrxiv/` - ChemRxivRetrieval results
- `results/ChEmbed/` - Benchmark results  
- `logs/` - Execution logs
