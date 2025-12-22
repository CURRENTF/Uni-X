from datasets import load_dataset

orca = load_dataset('../finevision', name='text_openorca', split='train', num_proc=32)

def convert_texts_to_conversations(sample):
    lst = []
    for turn in sample['texts']:
        lst.append({'from': 'human', 'value': turn['user']})
        lst.append({'from': 'assistant', 'value': turn['assistant']})
    return {"conversations": lst}

orca = orca.map(convert_texts_to_conversations, batched=False)
orca = orca.remove_columns(['images', 'texts', 'source', 'relevance_ratings', 'relevance_min', 'formatting_ratings', 'formatting_min'])
for _ in range(8):
    orca.shard(num_shards=8, index=_).to_parquet(f'../datasets/openorca/part-{_}.parquet')
