import subprocess
import sys

commands = [
    [sys.executable, "scripts/run_zero_nas.py", "--n-architectures", "5"],
    [sys.executable, "scripts/run_semantic_demo.py"],
    [sys.executable, "scripts/run_rulix_demo.py"],
]

for cmd in commands:
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)
