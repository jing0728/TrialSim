import json

with open('data/raw/filtered_studies.json', encoding='utf-8') as f:
    studies = json.load(f)

# 建一个 nct_id -> 原始文本的映射
raw_map = {}
for s in studies:
    mod = s.get('protocolSection', {})
    nct = mod.get('identificationModule', {}).get('nctId', '')
    txt = mod.get('eligibilityModule', {}).get('eligibilityCriteria', '')
    raw_map[nct] = txt

# 找出缺失 min_age 的 trial
with open('data/raw/parsed_pico.jsonl', encoding='utf-8') as f:
    data = [json.loads(l) for l in f]

missing = [d for d in data if d['demographics']['min_age'] == 'N/A']
print(f'Missing min_age: {len(missing)}/{len(data)}\n')

# 打印前8条的原始文本（只看前300字符）
for d in missing[:8]:
    nct = d['nct_id']
    raw = raw_map.get(nct, '')
    print(f"--- {nct} | {d['condition']}")
    print(f"  {raw[:300]}")
    print()