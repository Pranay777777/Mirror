import codecs
import json
import re

text = codecs.open('recal_logs_3.txt', 'r', 'utf-16le').read()

results = {"pos1": {}, "pos2right": {}}
current_video = None

for line in text.split('\n'):
    if 'Testing pos1' in line:
        current_video = "pos1"
    elif 'Testing pos2right' in line:
        current_video = "pos2right"
        
    if current_video and 'VAL_DIAG:' in line:
        match = re.search(r'VAL_DIAG: (.*?): ([\d\.\-]+)', line)
        if match:
            key, val = match.groups()
            results[current_video][key] = float(val)

print(json.dumps(results, indent=2))
