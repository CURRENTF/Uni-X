with open('dpg_prompts_en.jsonl', 'r', encoding='utf-8') as _in, open('dpg_prompts_en_fixed.jsonl', 'w') as _o:
    for l in _in:
        l = l.replace('prompts/', '').replace('.txt', '')
        _o.write(l)

with open('dpg_prompts_zh.jsonl', 'r', encoding='utf-8') as _in, open('dpg_prompts_zh_fixed.jsonl', 'w') as _o:
    for l in _in:
        l = l.replace('prompts/', '').replace('.txt', '')
        _o.write(l)
