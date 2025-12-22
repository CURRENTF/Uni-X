from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from torch import nn
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


def main(args):
    model_path = args.model_path
    vq_codebook_size = args.num_add_token

    llm = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    llm.generation_config.do_sample = True
    ori_vocabe_size = llm.config.vocab_size
    # 这个地方的 resize 是对的，但是训练使用的 token id 其实是不对的，占用了预留token，不过先不管了。
    llm.resize_token_embeddings(new_num_tokens=ori_vocabe_size + vq_codebook_size, mean_resizing=True)

    llm.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='/path/to/gemma-7b', help='path to ori model')
    parser.add_argument('--save_path', type=str, default='/path/to/save/gemma-7b-addtoken',
                        help='path to save new model')
    parser.add_argument('--num_add_token', type=int, default=8192, help='num of tokens need to be added')
    args = parser.parse_args()
    main(args)
