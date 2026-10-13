"""Bundle the repo into a Kaggle script kernel (script + kernel-metadata.json).

    python scripts/kaggle_bundle.py --slug hidden-vertex-part2-stage0 --title "Hidden Vertex Part2 Stage0" \
        --body scripts/kaggle/part2_stage0.py --out <dir>
    kaggle kernels push -p <dir> -t 3600
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_PREFIXES = ('results/', 'dataset/', 'notebooks/', 'docs/', '.venv/', '.git/')
USER = 'viveriun'
DATASET = 'atulchikku/lhc-olympics-2020-ad-r-and-d'

HEADER = '''import base64, glob, io, os, shutil, subprocess, sys, tarfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hdf5plugin", "torch-geometric"], check=False)
HV = "/kaggle/working/hv"
os.makedirs(HV, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__")), mode="r:gz").extractall(HV)
os.chdir(HV)
sys.path.insert(0, HV)
H5 = glob.glob("/kaggle/input/**/events_anomalydetection_v2.h5", recursive=True)[0]
FEATURES = (glob.glob("/kaggle/input/**/events_anomalydetection_v2.features.h5", recursive=True) or [None])[0]
OUT = "/kaggle/working/out"
CACHE = "/tmp/part2_cache"
os.makedirs(OUT, exist_ok=True)


def sh(*args, **env):
    """Run a command inside the extracted repo; raises on a non-zero exit."""
    subprocess.run([str(a) for a in args], cwd=HV, check=True, env=dict(os.environ, TQDM_DISABLE="1", **env))

'''
FOOTER = '\nshutil.rmtree(HV, ignore_errors=True)\n'


def _tracked_files(repo_root):
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=repo_root, check=True,
                         capture_output=True, text=True).stdout.splitlines()
    return [f for f in out if not f.startswith(EXCLUDE_PREFIXES) and os.path.isfile(os.path.join(repo_root, f))]


def build_script(body_text, repo_root=REPO):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for rel in _tracked_files(repo_root):
            tar.add(os.path.join(repo_root, rel), arcname=rel)
    payload = base64.b64encode(buffer.getvalue()).decode()
    return HEADER.replace('__PAYLOAD__', payload) + body_text + FOOTER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slug', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--body', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    script_name = f'{args.slug}.py'
    with open(args.body, encoding='utf-8') as handle:
        script = build_script(handle.read())
    with open(os.path.join(args.out, script_name), 'w', encoding='utf-8') as handle:
        handle.write(script)
    metadata = {
        'id': f'{USER}/{args.slug}', 'title': args.title, 'code_file': script_name, 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': True, 'keywords': ['gpu'], 'dataset_sources': [DATASET], 'kernel_sources': [],
        'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4',
    }
    with open(os.path.join(args.out, 'kernel-metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print(f'wrote {args.out}/{script_name} ({len(script) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
