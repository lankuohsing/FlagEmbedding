# 激活Conda环境的核心命令
source /root/anaconda3/etc/profile.d/conda.sh
conda activate py_310

export WANDB_MODE=disabled
home_dir="/mnt/workspace/languoxing"

train_data="${home_dir}/datasets/train_data_no_hn_dedup_202410_to_202503_lite.jsonl"
output_dir="${home_dir}/outputs/bge-base-zh-v1.5-finetuned_202410_202503_lite2"
model_name_or_path="${home_dir}/models/bge-base-zh-v1.5"
num_train_epochs=2
per_device_train_batch_size=128
num_gpus=2

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="${home_dir}/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path ${model_name_or_path} \
    --cache_dir $HF_HUB_CACHE \
"

# 修正1：内部双引号改为单引号
data_args="\
    --train_data $train_data \
    --cache_path ${home_dir}/.cache \
    --train_group_size 16 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval '为这个句子生成表示以用于检索相关文章： ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
"

# 修正2：--output_dir 引用变量
training_args="\
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --learning_rate 5e-6 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed /mnt/workspace/languoxing/projects/bge_embedder/ds_stage0.json \
    --logging_steps 1 \
    --logging_dir ${output_dir}/logs \
    --report_to 'tensorboard' \
    --save_steps 100 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
"

# 修正3：cmd 添加双引号防止参数分割错误
cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo "$cmd"
eval "$cmd"

