import json


lang = "zh"
filename = f"data_{lang}.json"

with open(filename, "r", encoding='utf-8') as f:
    data = json.load(f)

out = []
for item in data:
    if lang == 'en':
        tmp = {"query": item["prompt"], "unsafe_response": item["rejected"], "safe_response": item["chosen"]}
    elif lang == 'zh':
        tmp = {"query": item["question"], "unsafe_response": item["output"], "jb_prompt": item["prompt"]}
    out.append(tmp)

with open(f"new_{filename}", "w", encoding='utf-8') as f:
    f.write(json.dumps(out, indent=4, ensure_ascii=False))
