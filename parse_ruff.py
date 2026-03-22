import json

with open('ruff_out.json', 'r', encoding='utf-16') as f:
    data = json.load(f)


with open('ruff_clean.txt', 'w', encoding='utf-8') as f_out:
    for err in data:
        f_out.write(f"{err['filename']}:{err['location']['row']}:{err['location']['column']} - {err['code']} - {err['message']}\n")
