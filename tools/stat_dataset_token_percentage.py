import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, DatasetDict, IterableDatasetDict


# Mock the missing DataArguments class for standalone execution
class DataArguments:
    data_path: str = "../datasets/converted_data_shuffled"
    percentage: str = "1.0"
    shuffleseed: int = 42
    use_data_packing: bool = True
    streaming_data: bool = True
    vq_resolution: int = 512
    T2I_ratio: float = 0.5
    ignore_instruction: bool = True
    packing_bug_fix: bool = False
    add_sep_for_vis: bool = False
    vis_sep_tokens: str = "151935"
    vis_sep_lens: str = "32"


from uni_arch.train.train import make_supervised_data_module
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("../models/Qwen2.5-1.5B-AddTokens")
tok.model_max_length = 10240
# Use the mocked DataArguments class
data_args = DataArguments()
data = make_supervised_data_module(tok, data_args)

### TODO 添加统计一下使用了多少样本，这里只计算了token数目
# --- Start of my code ---

# create a dataloader from the dataset
# batch_size determines how many samples are passed to the collator at once
dataloader = DataLoader(
    data['train_dataset'],
    batch_size=10,
    collate_fn=data['data_collator'],
    # num_workers must be 0 for stateful collators in older PyTorch versions,
    # but can be > 0 if the collator and dataset are properly designed.
    # Using 8 as specified.
    num_workers=8,
)

# initialize counters
total_text_tokens = 0
total_vision_tokens = 0
total_samples = 0
num_batches_to_process = 1000
text_vocab_size = len(tok)

# iterate and count tokens
data_iterator = iter(dataloader)
for _ in tqdm(range(num_batches_to_process), desc="Processing batches"):
    try:
        batch = next(data_iterator)
        # the collator returns a tensor of shape [1, sequence_length]
        input_ids = batch['input_ids'].squeeze(0)

        # In packed datasets, EOS tokens separate individual samples.
        total_samples += torch.sum(input_ids == tok.eos_token_id).item() + 1

        # vision tokens are those with an ID greater than or equal to the text tokenizer's vocab size
        vision_mask = (input_ids >= text_vocab_size)

        total_vision_tokens += torch.sum(vision_mask).item()
        total_text_tokens += torch.sum(~vision_mask).item()

    except StopIteration:
        print(f"\nDataLoader exhausted after {_} batches.")
        num_batches_to_process = _ # Update the number of processed batches
        break
    except Exception as e:
        print(f"\nAn error occurred in batch {_}: {e}")
        continue

# calculate and print the final statistics
total_tokens = total_text_tokens + total_vision_tokens
if total_tokens > 0:
    text_ratio = total_text_tokens / total_tokens
    vision_ratio = total_vision_tokens / total_tokens

    print(f"\n--- Statistics over {num_batches_to_process} batches ---")
    print(f"Total samples processed: {total_samples}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"  - Text tokens:   {total_text_tokens} ({text_ratio:.2%})")
    print(f"  - Vision tokens: {total_vision_tokens} ({vision_ratio:.2%})")
    if total_vision_tokens > 0:
        print(f"Ratio of text to vision tokens: {total_text_tokens / total_vision_tokens:.2f} : 1")
else:
    print("No tokens were processed.")
# --- End of my code ---