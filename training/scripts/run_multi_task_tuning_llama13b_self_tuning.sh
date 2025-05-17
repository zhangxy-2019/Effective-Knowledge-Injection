#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# OUTPUT=$1
# ZERO_STAGE=$2
# if [ "$OUTPUT" == "" ]; then
#     OUTPUT=./llama2_7b_mc_5_shot_truthfulqa_train_sft_lre_7_epoch
# fi
# if [ "$ZERO_STAGE" == "" ]; then
#     ZERO_STAGE=3
# fi
# cpt=/apdcephfs_cq2/share_1603164/data/huggingface_models/Llama-2-7b-hf
LLAMA_CKPT_DIR=/models/Llama-2-13b-hf

## self-tuning stage 1

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/multi_task_stage1_epoch2
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/len14108bio_train_wiki_2023_10_multi_task_single.json \
   --data_eval_path /xyzhang/data/bio_domain/len14108bio_train_wiki_2023_10_multi_task_single.json \
   --model_name_or_path /models/Llama-2-13b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --print_loss \
   --stage 2 \
   --few_shot_prompts \
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log

## self-tuning stage 2

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/multi_task_stage1_epoch2_conso_epoch1
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_6517_train_qa_6136_test_multi_task_knowledge_consolidation.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_6517_train_qa_6136_test_multi_task_knowledge_consolidation.json \
   --model_name_or_path /xyzhang/cpts/llama2_13b/multi_task_stage1_epoch2/checkpoint-epoch1 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --print_loss \
   --stage 2 \
   --few_shot_prompts \
   --reload_hf_checkpoint \
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log

## self-tuning stage 3

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/multi_task_stage1_epoch2_conso_epoch1_test_epoch3
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /apdcephfs_cq10/share_160316/user/svetzhang/.xiaoying_folder/wiki_newpages/short_qa/bio_domain/wiki_2023_10_len_413_forget_prevent_train_qa_32_test_our_multi_task_single.json \
   --data_eval_path /apdcephfs_cq10/share_160316/user/svetzhang/.xiaoying_folder/wiki_newpages/short_qa/bio_domain/wiki_2023_10_len_413_forget_prevent_train_qa_32_test_our_multi_task_single.json \
   --model_name_or_path /xyzhang/cpts/llama2_13b/multi_task_stage1_epoch2_conso_epoch1/checkpoint-epoch0 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --print_loss \
   --stage 2 \
   --few_shot_prompts \
   --reload_hf_checkpoint \
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log


