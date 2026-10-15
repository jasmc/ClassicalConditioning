"""Run the independent, source-only technical protocol sensitivity."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from threadpoolctl import threadpool_limits
from classical_conditioning.analysis.inference.panel_g_technical import run_technical_followup

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    namespace=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-G-independent-review')
    if not args.output_dir.resolve().is_relative_to(namespace):raise ValueError('Use the dedicated external G namespace')
    with threadpool_limits(1):run_technical_followup(args.parent,args.output_dir)
