#!/bin/bash
set -e 

# example tasks:
tasks=(
  wmdp_bio
  wmdp_bio_rephrased_english_filler
  wmdp_bio_rephrased_hindi_filler
  wmdp_bio_rephrased_latin_filler
  wmdp_bio_rephrased_conversation
  wmdp_bio_rephrased_poem
  wmdp_bio_rephrased_replace_with_variables
  wmdp_bio_rephrased_technical_terms_removed_1
  wmdp_bio_rephrased_translated_farsi
  wmdp_bio_rephrased_translated_german
  wmdp_bio_rephrased_translated_korean
)
tasks_list=$(printf "%s," "${tasks[@]}" | sed 's/,$//')

# example models:
models=(
  "cais/Zephyr_RMU"
  "ScaleAI/mhj-llama3-8b-rmu"
  "lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2"
  "baulab/elm-zephyr-7b-beta"
  "baulab/elm-Mistral-7B-v0.1"
  "baulab/elm-Meta-Llama-3-8B"
  "LLM-GAT/llama-3-8b-instruct-graddiff-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-elm-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-tar-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-rr-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-rmu-checkpoint-8"
  "LLM-GAT/llama-3-8b-instruct-rmu-lat-checkpoint-8"
)

# code for Table 1 & 2
for model in "${models[@]}"; do
  for task in "${tasks[@]/#wmdp_bio_rephrased_/}"; do
    echo "Evaluating $model on $task..."
    python3 evals/inference.py --ckpt_dir "$model" --data_dir "./wmdp_rephrased/data_${task}/test/" --dataset_name "bio_questions"
  done
done

# code for Table 3
for model in "${models[@]}"; do
  echo "Evaluating $model..."
  lm_eval --model hf \
    --model_args pretrained="$model",dtype="bfloat16" \
    --tasks $tasks_list
done