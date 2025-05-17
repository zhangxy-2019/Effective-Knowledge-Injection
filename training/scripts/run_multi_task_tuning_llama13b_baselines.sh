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
# cpt=/huggingface_models/Llama-2-7b-hf
LLAMA_CKPT_DIR=/models/Llama-2-13b-hf

## continued pre-training
ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/conti_train_test_epoch5
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_127_num_qa_0_train_doc_0_test_doc_127_doc_test.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_127_num_qa_0_train_doc_0_test_doc_127_doc_test.json \
   --model_name_or_path /models/Llama-2-13b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 5 \
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

## mixed training

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/mix_train_epoch3
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_7399_num_qa_6136_train_doc_1136_test_doc_127_mix_train.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_7399_num_qa_6136_train_doc_1136_test_doc_127_mix_train.json \
   --model_name_or_path /models/Llama-2-13b-hf \
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
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log

## PIT training

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/pit_add_seq_stage1_epoch1
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_6136_num_qa_6136_train_doc_0_test_doc_0_qa_train.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_6136_num_qa_6136_train_doc_0_test_doc_0_qa_train.json \
   --model_name_or_path /models/Llama-2-13b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
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
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log


ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/pit_add_seq_stage1_epoch3
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_sequential_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_12272_num_qa_6136_train_doc_1136_test_doc_0_pit_seq_train.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_12272_num_qa_6136_train_doc_1136_test_doc_0_pit_seq_train.json \
   --model_name_or_path /xyzhang/cpts/llama2_13b/pit_add_seq_stage1_epoch1/checkpoint-epoch0 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
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
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/pit_add_seq_stage1_epoch3_stage2_epoch3
mkdir -p $OUTPUT_DIR1 

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_159_num_qa_32_train_doc_0_test_doc_127_forget_prevent_qa_train32_test.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_159_num_qa_32_train_doc_0_test_doc_127_forget_prevent_qa_train32_test.json \
   --model_name_or_path /xyzhang/cpts/llama2_13b/pit_add_seq_stage1_epoch3/checkpoint-epoch2 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
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
   --reload_hf_checkpoint \
   --few_shot_prompts \
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log


## standard instruction-tuning

ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/stand_tune_stage1_epoch3
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_1263_num_qa_0_train_doc_1136_test_doc_127_doc_standard_train.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_1263_num_qa_0_train_doc_1136_test_doc_127_doc_standard_train.json \
   --model_name_or_path /models/Llama-2-13b-hf \
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
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log



ZERO_STAGE=3
OUTPUT_DIR1=/xyzhang/cpts/llama2_13b/stand_tune_stage1_epoch3_stage2_epoch1
mkdir -p $OUTPUT_DIR1

deepspeed --master_port 12351 main_PIT_train.py \
   --data_path /xyzhang/data/bio_domain/wiki_2023_10_len_6136_num_qa_6136_train_doc_0_test_doc_0_qa_train.json \
   --data_eval_path /xyzhang/data/bio_domain/wiki_2023_10_len_6136_num_qa_6136_train_doc_0_test_doc_0_qa_train.json \
   --model_name_or_path /xyzhang/cpts/llama2_13b/stand_tune_stage1_epoch3/checkpoint-epoch2 \
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
   --reload_hf_checkpoint \
   --few_shot_prompts \
   --instruction_prompts \
   --save_steps 250 \
   --output_dir $OUTPUT_DIR1 \
   &> $OUTPUT_DIR1/training.log
