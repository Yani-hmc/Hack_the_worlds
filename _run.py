import sys
from examples.eeg.main import run
kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
run(**kw)
