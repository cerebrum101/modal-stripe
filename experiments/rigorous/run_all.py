"""Run the audit in dependency order with evidence archives before/after."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'final_evaluations.json').exists():subprocess.run([sys.executable,str(ROOT/'archive.py')],check=True)
for name in ('study.py','measure.py','calibrate.py','local_control.py','exact_window_audit.py','audit_data.py','test_framework.py','verify_results.py','report.py','archive.py'):
 subprocess.run([sys.executable,str(ROOT/name)],check=True)
