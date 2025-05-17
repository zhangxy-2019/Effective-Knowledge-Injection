#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
LLAMA_CKPT_DIR=models/Llama-2-13b-hf

## inference

PROMPTS_DIR=/data/hard_data/wiki_2024_5_shot_test_short_qa_prompts.json
DATA_DIR=/data/hard_data/wiki_2023_10_len_663_open_book_test.json
# DATA_DIR=/data/hard_data/wiki_2023_9_10_film_len_955_open_book_test.json
# DATA_DIR=/data/hard_data/wiki_2023_10_len_1502_openbook_test.json
PARAM_SIZE=7 # 7, 13, 33, 65
MODEL_TYPE=llama
LLAMA_CKPT_DIR_PREFIXES=(
    "xyzhang/cpts/checkpoint-epoch2"

)
for i in "${!LLAMA_CKPT_DIR_PREFIXES[@]}"; do
    LLAMA_CKPT_DIR=${LLAMA_CKPT_DIR_PREFIXES[$i]}
    # OUTPUT_DIR=${LLAMA_CKPT_DIR}/tfqa
    OUTPUT_DIR=${LLAMA_CKPT_DIR}/eval
    # rest of the script
    python3 wiki_vllm_inference.py \
        --model_path ${LLAMA_CKPT_DIR} \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --temperature 1.0 \
        --top_k 10 \
        --top_p 0.9 \
        --num_devices 8 \
        --prompts_dir ${PROMPTS_DIR} \
        --max_prompt_length 2048 \
        --num_sampling 1 \
        --few_shot
done

## memorization eval on wiki-newpages-2023-qa

DATA_DIR=/data/hard_data/wiki_2023_10_len_127_num_qa_0_train_doc_0_test_doc_127_doc_test.json
LLAMA_CKPT_DIR_PREFIXES=(
    "xyzhang/cpts/checkpoint-epoch2"
)
PARAM_SIZE=13 # 7, 13, 33, 65
MODEL_TYPE=llama
EVAL_MODE=gen
# EVAL_MODE=train
for i in "${!LLAMA_CKPT_DIR_PREFIXES[@]}"; do
    LLAMA_CKPT_DIR=${LLAMA_CKPT_DIR_PREFIXES[$i]}
    OUTPUT_DIR=${LLAMA_CKPT_DIR}/ppl_eval

    python3 vllm_ppl_new_eval.py \
        --model_path ${LLAMA_CKPT_DIR} \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --temperature 1.0 \
        --top_k 10 \
        --top_p 0.9 \
        --num_devices 8 \
        --max_prompt_length 2048
done

## extraction eval on wiki-newpages-2023-qa

# PROMPT_DIR=/data/hard_data/wiki_2023_10_len_1502_openbook_test.json
PROMPT_DIR=/data/hard_data/wiki_2023_10_len_663_open_book_test.json
# PROMPT_DIR=/data/hard_data/wiki_2023_9_10_film_len_955_open_book_test.json
LLAMA_CKPT_DIR_PREFIXES=(
    "xyzhang/cpts/checkpoint-epoch2"
)
PARAM_SIZE=7 # 7, 13, 33, 65
MODEL_TYPE=llama
EVAL_MODE=gen
# EVAL_MODE=train
for i in "${!LLAMA_CKPT_DIR_PREFIXES[@]}"; do
    LLAMA_CKPT_DIR=${LLAMA_CKPT_DIR_PREFIXES[$i]}
    DATA_DIR=${LLAMA_CKPT_DIR}/eval/run_results_llama_7b_temp_1.0_top_p_0.9_top_k_10_len_663_few_shot_True_openbook_False_prompting.json
    OUTPUT_DIR=${LLAMA_CKPT_DIR}/eval
    # rest of the script
    python3 eval_analyze.py \
        --question_dir ${PROMPT_DIR} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --eval_mode ${EVAL_MODE}
done

## reasoning eval on wiki-newpages-2023-qa

DATA_DIR=/data/hard_data/wiki_bio_test_doc_127_test_qa_729_nli.json
LLAMA_CKPT_DIR_PREFIXES=(
    "xyzhang/cpts/checkpoint-epoch2"
)
PARAM_SIZE=7 # 7, 13, 33, 65
MODEL_TYPE=llama

for i in "${!LLAMA_CKPT_DIR_PREFIXES[@]}"; do
    LLAMA_CKPT_DIR=${LLAMA_CKPT_DIR_PREFIXES[$i]}
    OUTPUT_DIR=${LLAMA_CKPT_DIR}/nli_eval
    # rest of the script
    python3 csqa_vllm_inference.py \
        --model_path ${LLAMA_CKPT_DIR} \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --temperature 1.0 \
        --top_k 10 \
        --top_p 0.9 \
        --num_devices 8 \
        --max_prompt_length 2048 \
        --num_sampling 1 \
        --few_shot
done

## knowledge retention (extraction) eval on natural questions

DATA_DIR=/data/nq_test_len1769_data.json
LLAMA_CKPT_DIR_PREFIXES=(
    "xyzhang/cpts/checkpoint-epoch2"
)
PARAM_SIZE=7 # 7, 13, 33, 65
MODEL_TYPE=llama

for i in "${!LLAMA_CKPT_DIR_PREFIXES[@]}"; do
    LLAMA_CKPT_DIR=${LLAMA_CKPT_DIR_PREFIXES[$i]}
    OUTPUT_DIR=${LLAMA_CKPT_DIR}/nq_eval
    # rest of the script
    python3 nq_vllm_inference.py \
        --model_path ${LLAMA_CKPT_DIR} \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --temperature 1.0 \
        --top_k 10 \
        --top_p 0.9 \
        --num_devices 8 \
        --max_prompt_length 2048 \
        --num_sampling 1 \
        --few_shot
done

## knowledge retention (reasoning) eval on commonsense qa

DATA_DIR=/data/mc_5_shot_1221_dev_mc.json
LLAMA_CKPT_DIR_PREFIXES=(
    "xyzhang/cpts/checkpoint-epoch2"
)
PARAM_SIZE=7 # 7, 13, 33, 65
MODEL_TYPE=llama

for i in "${!LLAMA_CKPT_DIR_PREFIXES[@]}"; do
    LLAMA_CKPT_DIR=${LLAMA_CKPT_DIR_PREFIXES[$i]}
    OUTPUT_DIR=${LLAMA_CKPT_DIR}/csqa_eval
    # rest of the script
    python3 csqa_vllm_inference.py \
        --model_path ${LLAMA_CKPT_DIR} \
        --param_size ${PARAM_SIZE} \
        --model_type ${MODEL_TYPE} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --temperature 1.0 \
        --top_k 10 \
        --top_p 0.9 \
        --num_devices 8 \
        --max_prompt_length 2048 \
        --num_sampling 1 \
        --few_shot
done



# DATA_DIR=/data/hard_data/wiki_bio_test_doc_127_test_qa_729_nli.json
# LLAMA_CKPT_DIR=models/Llama-2-13b-hf
# PARAM_SIZE=13 # 7, 13, 33, 65
# MODEL_TYPE=llama
# EVAL_MODE=gen
# # EVAL_MODE=train

# OUTPUT_DIR=xyzhang/cpts/llama2_13b/baseline/nli_eval

# python3 csqa_vllm_inference.py \
#     --model_path ${LLAMA_CKPT_DIR} \
#     --param_size ${PARAM_SIZE} \
#     --model_type ${MODEL_TYPE} \
#     --data_dir ${DATA_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --temperature 1.0 \
#     --top_k 10 \
#     --top_p 0.9 \
#     --num_devices 8 \
#     --max_prompt_length 2048 \
#     --num_sampling 1 \
#     --few_shot

# DATA_DIR=/data/hard_data/wiki_2023_10_len_127_num_qa_0_train_doc_0_test_doc_127_doc_test.json
# PARAM_SIZE=13 # 7, 13, 33, 65
# MODEL_TYPE=llama
# OUTPUT_DIR=xyzhang/cpts/llama2_13b/baseline/ppl_eval
#     # rest of the script
# python3 vllm_ppl_new_eval.py \
#     --model_path ${LLAMA_CKPT_DIR} \
#     --param_size ${PARAM_SIZE} \
#     --model_type ${MODEL_TYPE} \
#     --data_dir ${DATA_DIR} \
#     --output_dir ${OUTPUT_DIR} \
#     --temperature 1.0 \
#     --top_k 10 \
#     --top_p 0.9 \
#     --num_devices 8 \
#     --max_prompt_length 2048





