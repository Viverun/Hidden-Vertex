import base64
import io
import os
import re
import tarfile

from scripts.kaggle_bundle import build_script

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_bundle_contains_source_and_excludes_data_dirs():
    script = build_script("print('body')", ROOT)
    payload = re.search(r'b64decode\("([A-Za-z0-9+/=]+)"\)', script).group(1)
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz').getnames()
    assert 'src/benchmarks/report.py' in names and 'scripts/part2_run.py' in names
    assert not any(n.startswith(('results/', 'dataset/', 'notebooks/', '.git/', '.venv/')) for n in names)
    assert script.rstrip().endswith("shutil.rmtree(HV, ignore_errors=True)")
    assert "print('body')" in script
    compile(script, 'kernel.py', 'exec')
