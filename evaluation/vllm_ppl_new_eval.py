# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
from vllm import LLM, SamplingParams
import json 
# from transformers import (
#     AutoModelForCausalLM, )
from transformers.models.llama import LlamaTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
import json
import os
import sys
import time 
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM
# import tensor_parallel as tp
import accelerate

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from utils.model.model_utils import create_hf_model
# from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help='Prompt data path',
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help='Output path',
        required=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help='Specify temperature for sampling',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=8,
        help='Specify num of gpus',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help='Specify num of return sequences',
    )
    # parser.add_argument(
    #     "--num_sampling",
    #     type=int,
    #     default=30,
    #     help='Specify num of return sequences',
    # )
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--openbook', action='store_true')
    args = parser.parse_args()

    return args

def load(ckpt_dir, model_type):
    # n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # tokenizer = LlamaTokenizer.from_pretrained(
    #     ckpt_dir,
    #     use_fast=False,
    #     padding_side="left",
    # )
    # tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    # tokenizer.bos_token_id = 1
    
    # if model_type == 'llama':
    #     # we use tensor parallel for loading llama
    #     model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    #     model = tp.tensor_parallel(model, [i for i in range(n_gpus)]) 
    # else:
    model_config = AutoConfig.from_pretrained(ckpt_dir)
    model = LlamaForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float16, config=model_config)
    # model.to(device)
    # model.eval()

    return model, tokenizer

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t]

    return input_tokens

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts):
    batch_size = 1
    loss_list = []
    ppl_list = []
    losses = 0
    ppls  = 0
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        print("ende_inputs: ", encode_inputs)
        encode_inputs["input_ids"] = encode_inputs["input_ids"].to('cuda:0')
        with torch.no_grad():
            outputs = model(input_ids=encode_inputs["input_ids"], labels=encode_inputs["input_ids"])
            print("encode_inputs: ", encode_inputs["input_ids"])
        loss = outputs.loss
        print("loss", loss)
        perplexity = torch.exp(loss)
        perplexity = perplexity.detach().cpu().numpy()
        ppl_list.append(perplexity)
        ppls += perplexity
        losses += loss
        loss_list.append(loss.detach().cpu().numpy())
    
    losses = losses.detach().cpu().numpy()
    losses = losses / len(prompts)
    ppls = ppls / len(prompts)

    # try:
    #     losses_tensor = torch.tensor(losses, dtype=torch.float32)  # Convert losses back to a tensor
    #     perplexity = torch.exp(losses_tensor)
    #     perplexity = perplexity.detach().cpu().numpy()
    # except OverflowError:
    #     perplexity = float("inf")
    print(ppls)
    return ppls, loss_list, ppl_list

# def batch_infer(model, tokenizer, prompts):
#     batch_size = 8
#     answers = []
#     for batch_input in tqdm(batch_split(prompts, batch_size)):
#         encode_inputs = prepare_input(tokenizer, batch_input)
#         outputs = model.generate(**encode_inputs, max_new_tokens=1)
#         answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
#     answers = [answer[-1] for answer in answers]
#     return answers

def main():
    args = parse_args()

    # # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    # tokenizer = LlamaTokenizer.from_pretrained(args.model_path, fast_tokenizer=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    model, tokenizer = load(args.model_path, args.model_type)
    model = model.to('cuda:0')
    # run_results = {}
    df_data = json.load(open(args.data_dir))
    prompt_data = df_data["instances"]
    print('Inferencing ...')
    prompts = [record['output'] for record in prompt_data]
    print("len prompts: ", len(prompts))
    # gold_answers = [record['answer'] for record in records]
    ppls, loss_list, ppl_list = batch_infer(model, tokenizer, prompts)
    ppls = ppls.tolist()  # If ppl is th
    ppl_list = [ppl.tolist() for ppl in ppl_list]  # If ppl is the ndarray
    loss_list = [loss.tolist() for loss in loss_list]  # If loss_list is a list of ndarrays
    print("ppl_answers: ", ppls) 
    run_results = {"ppl_answers": ppls, "loss_list": loss_list, "pple_list": ppl_list}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_filename = 'run_results_ppl_single_calculation.json'
    output_filename = os.path.join(args.output_dir, output_filename)
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    # Free up GPU memory
    # torch.cuda.empty_cache()
    # compute_metric(output_filename, args.output_dir)
    

if __name__ == "__main__":
    main()
