# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import string
import re
import argparse
from collections import Counter
# import spacy
# spacy.load('en_core_web_sm')
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# from vllm import LLM, SamplingParams
import json 
# from transformers import (
#     AutoModelForCausalLM, )
from tqdm import tqdm
from transformers.models.llama import LlamaTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# from utils.model.model_utils import create_hf_model
# from utils.utils import load_hf_tokenizer
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--data_dir",
        type=str,
        help='Prompt data path',
        required=True,
    )
    parser.add_argument(
        "--question_dir",
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
        "--eval_mode",
        type=str,
        default='mc',
        required=True,
    )
    args = parser.parse_args()

    return args

FALSE_RESPONSES = ["is unknown",
                   "The answer is uncertain",
                   "The answer is unclear",
                   "I have no idea",
                   "have no idea",
                   "I'm not sure",
                   "I am not sure",
                   "I don’t know",
                   "There is no concrete answer to this question",
                   "It is impossible to answer",
                   "There is no known case",
                   "There is no public information available",
                   "There is no scientific evidence",
                   "There is no right answer",
                   "It is impossible to know",
                   "It is difficult to predict",
                   ]


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print("f1: ", f1)
    # print("recall: ", recall)
    return f1, recall

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


def compute_train_metric(output_filename, output_dir):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    eval_result = []
    raw_result = []
    sim_scores = []
    recall_list = []
    em_list = []
    f1_list = []
    rouge_l = 0
    fmeasure_scores = [] # rouge_l
    pred_answers = run_results['pred_answers']
    gold_answers = run_results['gold_answers']
    for pred, gold in tqdm(zip(pred_answers, gold_answers)):
        pred = pred.strip().lower()
        gold = gold.strip().lower()
        # if pred.endswith("."):
        #     pred = pred[:-1]
        # if gold.endswith("."):
        #     gold = gold[:-1]
        # recall_score = 0
        # if gold.lower() in pred.lower():
        #     recall_score = 1
        # recall_list.append(recall_score)
        em_list.append(exact_match_score(pred, gold))
        f1_sco, recall_score = f1_score(pred, gold)
        recall_list.append(recall_score)
        f1_list.append(f1_sco)
        ## compute fmeasure of rougel
        scores = scorer.score(gold.lower(), pred.lower())
        fmeasure_scores.append(scores['rougeL'].fmeasure)
        raw_result.append({"pred": pred, "gold": gold, "em_score": exact_match_score(pred, gold), "f1_score":f1_sco,  "recall": recall_score, "rougel": scores['rougeL'].fmeasure})
    # print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
    # total_acc += acc
    total_num = len(gold_answers)
    total_recall = sum(recall_list)/total_num
    eval_result.append({"recall": total_recall})
    eval_result.append({"EM": sum(em_list)/len(gold_answers)})
    eval_result.append({"f1_score": sum(f1_list)/len(gold_answers)})
    eval_result.append({"rouge_l": rouge_l})
    eval_result.append({"raw_evaluation": raw_result})
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(eval_result, open(os.path.join(output_dir, "train_eval_results.json"),'w'),indent=2)  

def compute_metric(output_filename, question_dir, output_dir, model, tokenizer):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_em = 0
    total_f1 = 0
    total_pre = 0
    total_acnt = 0
    eval_result = []
    raw_result = []
    sim_scores = []
    recall_list = []
    em_list = []
    f1_list = []
    rouge_l = 0
    fmeasure_scores = [] # rouge_l
    pred_answers = run_results['pred_answers']
    gold_answers = run_results['gold_answers']
    questions = json.load(open(question_dir))['instances']
    questions = [item["text"] for item in questions]

    for pred, gold, prompt in tqdm(zip(pred_answers, gold_answers, questions)):
        pred = pred.strip().lower()
        gold = gold.strip().lower()
        if pred.endswith("."):
            pred = pred[:-1]
        if gold.endswith("."):
            gold = gold[:-1]
        sim_score = 0
        # recall_score = 0
        # if gold.lower() in pred.lower():
        #     recall_score = 1
        # recall_list.append(recall_score)
        em_list.append(exact_match_score(pred, gold))
        # for false_response in FALSE_RESPONSES:
        #     if false_response.lower() in pred.lower():
        #         sim_score = 2
        if sim_score == 0: # string matching
            if gold.lower() == pred.lower(): 
                sim_score = 1
            else:
                sim_score = compute_similarity(pred, gold, prompt, tokenizer, model)
                # sim_score = compute_similarity_noprompt(pred, gold, tokenizer, model)
        sim_scores.append(sim_score)
        # acc += sim_score
        ## compute fmeasure of rougel
        f1_sco, recall_score = f1_score(pred, gold)
        recall_list.append(recall_score)
        f1_list.append(f1_sco)
        scores = scorer.score(gold.lower(), pred.lower())
        fmeasure_scores.append(scores['rougeL'].fmeasure)
        raw_result.append({"pred": pred, "gold": gold, "em_score": exact_match_score(pred, gold), "f1_score":f1_sco,  "eval_result": sim_score, "recall": recall_score, "rougel": scores['rougeL'].fmeasure})
    # print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
    # total_acc += acc
    total_num = len(gold_answers)
    total_acc = sim_scores.count(1) / total_num
    # print("ACC-all: %.4f" % (total_acc/total_num))
    # total_pre = sim_scores.count(1) / (sim_scores.count(2) + sim_scores.count(1))
    # total_acnt = 1 - sim_scores.count(0)/ total_num
    # alpha = 1 - sim_scores.count(2)/ total_num ## answer rate
    # rely = alpha * total_acnt + (1- alpha) * total_acc ## reliability score
    rouge_l = sum(fmeasure_scores)/len(fmeasure_scores)
    total_recall = sum(recall_list)/total_num
    eval_result.append({"accuracy": total_acc})
    eval_result.append({"recall": total_recall})
    eval_result.append({"EM": sum(em_list)/len(gold_answers)})
    eval_result.append({"f1_score": sum(f1_list)/len(gold_answers)})
    # eval_result.append({"answer_rate": alpha})
    # eval_result.append({"reliability": rely})
    eval_result.append({"rouge_l": rouge_l})
    eval_result.append({"raw_evaluation": raw_result})
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(eval_result, open(os.path.join(output_dir, "eval_results.json"),'w'),indent=2)  

def compute_mc_metric(output_filename, output_dir):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    eval_result = []
    raw_result = []
    for task in run_results:
        acc = 0
        pred_answers = [x.split(". ")[0] for x in run_results['pred_answers']]
        gold_answers = [x.split(". ")[0] for x in run_results['gold_answers']]
        for pred, gold in zip(pred_answers, gold_answers):
            sim_score = 0
            if pred.strip() == gold.strip(): 
                acc += 1
                sim_score = 1
            raw_result.append({"pred": pred, "gold": gold, "eval_result": sim_score})
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))
    acc = total_acc/total_num
    eval_result.append({"accuracy": acc})
    eval_result.append({"raw_evaluation": raw_result})
    json.dump(eval_result, open(os.path.join(output_dir, "vllm_eval_results.json"),'w'),indent=2)  

def compute_similarity_noprompt(pred, gold, tokenizer, model):
    # prompt = prompt.strip().lower()
    # print("prompt: ", prompt)
    input = pred.strip().lower() + ' [SEP] ' + gold.strip().lower() 
    # print("input: ", input)
    encoded_input = tokenizer.encode(input, padding=True)    
    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda:0'))['logits']
    predicted_label = torch.argmax(prediction, dim=1)

    reverse_input =  gold.strip().lower() + ' [SEP] ' + pred.strip().lower()
    # print("reverse_input: ", reverse_input)
    reverse_encoded_input = tokenizer.encode(reverse_input, padding=True)
    reverse_prediction = model(torch.tensor(torch.tensor([reverse_encoded_input]), device='cuda:0'))['logits']
    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
    if (2 in predicted_label) and (2 in reverse_predicted_label):
        return 1
    else:
        return 0
    
def compute_similarity(pred, gold, prompt, tokenizer, model):
    prompt = prompt.strip().lower()
    # print("prompt: ", prompt)
    input = prompt + pred.strip().lower() + ' [SEP] ' + prompt + gold.strip().lower() 
    # print("input: ", input)
    encoded_input = tokenizer.encode(input, padding=True)    
    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda:0'))['logits']
    predicted_label = torch.argmax(prediction, dim=1)

    reverse_input = prompt + gold.strip().lower() + ' [SEP] ' + prompt + pred.strip().lower()
    # print("reverse_input: ", reverse_input)
    reverse_encoded_input = tokenizer.encode(reverse_input, padding=True)
    reverse_prediction = model(torch.tensor(torch.tensor([reverse_encoded_input]), device='cuda:0'))['logits']
    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
    if (2 in predicted_label) and (2 in reverse_predicted_label):
        return 1
    else:
        return 0
    # if (0 not in predicted_label) and (0 not in reverse_predicted_label):
    #     return 1
    # else:
    #     return 0
def main():

    args = parse_args()
    device_id = 0
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')


    if args.eval_mode == "mc":
        compute_mc_metric(args.data_dir, args.output_dir)
    elif args.eval_mode == "train":
        compute_train_metric(args.data_dir, args.output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("/models/microsoft/deberta-large-mnli")
        model = AutoModelForSequenceClassification.from_pretrained("/models/microsoft/deberta-large-mnli")
        model.to(device)
        compute_metric(args.data_dir, args.question_dir, args.output_dir, model, tokenizer)
        # compute_metric(args.data_dir, args.output_dir, model, tokenizer)

if __name__ == "__main__":
    main()
