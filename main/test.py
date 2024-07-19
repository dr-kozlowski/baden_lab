import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.experiment_info import load_json

fpath = 'experiment_info\zf_leds_for_analysis.json'
print(load_json(fpath))