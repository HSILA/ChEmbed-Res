# MTEB Evaluation Scripts

## Setup
```bash
source .venv/bin/activate
uv sync
```

## Usage
```bash
python chemrxiv_bench.py   # 52 models × ChemRxivRetrieval
python nomic_bench.py      # 6 models × MTEB(eng,v2) + ChemTEB
```

## Output
- `results/chemrxiv/` - ChemRxivRetrieval results
- `results/ChEmbed/` - Benchmark results  
- `logs/` - Execution logs
