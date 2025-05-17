# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import argparse
import logging
import string
import re
import argparse
from collections import Counter
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
from vllm import LLM, SamplingParams
import json 
# from transformers import (
#     AutoModelForCausalLM, )
from transformers.models.llama import LlamaTokenizer
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
    parser.add_argument(
        "--num_sampling",
        type=int,
        default=30,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--chat",
        action='store_true'
    )
    parser.add_argument(
        "--openbooking",
        action='store_true'
    )
    parser.add_argument(
        "--few_shot",
        action='store_true'
    )
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)

    args = parser.parse_args()

    return args

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    print("scores for ground truths: ", scores_for_ground_truths)
    return max(scores_for_ground_truths)

def has_exact_match(ground_truths, candidates):
    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False

def get_oracle_score(ground_truth, predicted_answers, output_dir):
    exact_match = common = 0
    for gt, pred in zip(ground_truth, predicted_answers):
        # prediction = normalize_answer(pred)
        # em_for_this_question = has_exact_match(gt, prediction)
        em_for_this_question = exact_match_score(gt, pred)
        exact_match += int(em_for_this_question)

    exact_match = 100.0 * exact_match / len(ground_truth)

    results = {'oracle_exact_match': exact_match, 'common': common,
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(results, open(os.path.join(output_dir, "oracle_eval_results.json"),'w'),indent=2)  

def evaluate_triviaqa(ground_truth, predicted_answers, output_dir):
    f1 = exact_match =0
    for gt, pred in zip(ground_truth, predicted_answers):
        prediction = pred
        ground_truths = gt
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(predicted_answers)
    f1 = 100.0 * f1 / len(predicted_answers)
    results = {'exact_match': exact_match, 'f1': f1,
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(results, open(os.path.join(output_dir, "eval_results.json"),'w'),indent=2)  

def prepare_sample_text(input):
    """Prepare the text from a sample of the dataset."""
    text = f"[INST]\n{input}\n\n[/INST]"
    return text


def main():
    args = parse_args()
    # if args.few_shot:
    # few_shot_prompts = json.load(open(args.prompts_dir))[0]

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=200, stop=["\n\n", "Question:"])
    # Create an LLM.
    # stop=['\n\n', '\n', "Question:"]
    llm = LLM(model=args.model_path, tensor_parallel_size=args.num_devices)
    run_results = {}
    df_data = json.load(open(args.data_dir))
    prompt_data = df_data["instances"]
    # prompt_data = [item for item in prompt_data if item["task"]== "instruction_following"]
    print('Inferencing ...')
    records = []
    for prompt_data in prompt_data:
        prompt = prompt_data["text"]
        print("prompt: ", prompt)
        if args.chat:
            prompt = prepare_sample_text(prompt)
        label = prompt_data["output"]
        if args.few_shot:
            while len(tokenizer.tokenize(prompt)) + 1> args.max_prompt_length: # bos token
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)
        print(f"*** prompt: {prompt}, answer: {label} ***")
        recor_dict = {'prompt':prompt, 'answer':label}
        recor_dict_list = [recor_dict] * args.num_sampling
        records += recor_dict_list 

    prompts = [record['prompt'] for record in records]
    gold_answers = [record['answer'] for record in records]

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    pred_answers = []
    # pred_cum_logprobs = []

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # pred_cum_logprob = output.outputs[0].cumulative_logprob
        pred_answers.append(generated_text)
        # pred_cum_logprobs.append(pred_cum_logprob)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r},  Generated cum_logporbs: {str(pred_cum_logprob)!r}")

    run_results = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    # run_results = {'pred_answers':pred_answers, 'pred_cum_logprobs': pred_cum_logprobs, 'gold_answers':gold_answers}
    # run_results['gold_answers'], run_results['pred_answers']
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_filename = 'run_results_%s_%sb_temp_%s_top_p_%s_top_k_%s_len_%s_few_shot_%s_nq_%s_prompting.json' % (args.model_type, args.param_size, args.temperature, args.top_p, args.top_k, str(len(pred_answers)), str(args.few_shot), str(args.openbooking))
    output_filename = os.path.join(args.output_dir, output_filename)
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    # run_results = json.load(open('/apdcephfs_cq10/share_1603164/user/svetzhang/.xiaoying_folder/wiki_newpages/short_qa/short_qa_results/multi_task_single_train_stage2_no_forget128/checkpoint-epoch2/csqa_eval/run_results_llama_7b_temp_1.0_top_p_0.9_top_k_10_len_1221_few_shot_True_nq_False_prompting.json'))
    evaluate_triviaqa(run_results['gold_answers'], run_results['pred_answers'], args.output_dir)
    # get_oracle_score(run_results['gold_answers'], run_results['pred_answers'], args.output_dir)
    # output_filename = 'run_results_%s_%sb_temp_%s_top_p_%s_top_k_%s_len_%s_few_shot_%s_nq_%s_prompting.json' % (args.model_type, args.param_size, args.temperature, args.top_p, args.top_k, str(len(pred_answers)), str(args.few_shot), str(args.openbooking))
    # output_filename = os.path.join(args.output_dir, output_filename)
    # with open(output_filename, 'w') as f:
    #     json.dump(run_results, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    main()
