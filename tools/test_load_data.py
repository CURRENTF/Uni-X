from datasets import load_dataset
import glob
from tqdm import tqdm
from datasets import Features, List, Value

FEATURES = Features({
    "id": Value("string"),
    "text": Value("string"),
    "score": Value("float"),
    "meta": Features({"redpajama_set_name": Value("string")}),
    "data_type": Value("string"),
    "caption": Value("string"),
    "vqcode_512": Value("string"),
    "vqcode_multi768": Value("string"),
    "conversations": List(Features({"from": Value("string"), "value": Value("string")})),
    "height": Value("string"),
    "width": Value("string"),
    "length": Value("float64"),
    "metadata": Features({
        "date": Value("timestamp[us]"),
        "dump": Value("string"),            # 数据来源
        "file_path": Value("string"),       # 文件路径
        "int_score": Value("int64"),        # 整数分数
        "language": Value("string"),        # 语言
        "language_score": Value("float64"), # 语言得分
        "score": Value("float64"),          # 分数
        "token_count": Value("int64"),      # token 数量
        "url": Value("string"),             # 来源 url
    })
})

files = glob.glob('../datasets/uni_ct_v2/*')
# d = load_dataset('parquet', data_files=files, split='train', num_proc=32, features=FEATURES)

for i in tqdm(range(0, len(files), 32)):
    sub_files = files[i:i+32]
    try:
        d = load_dataset('parquet', data_files=sub_files, split='train', num_proc=32, features=FEATURES)
    except:
        for f in sub_files:
            try:
                _d = load_dataset('parquet', data_files=[f], split='train')
            except:
                print(f"error {f}")
