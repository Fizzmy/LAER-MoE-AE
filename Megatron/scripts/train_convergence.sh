export CUDA_DEVICE_MAX_CONNECTIONS=1 # to enable CP computation/communication streams to overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # to avoid max_reserved_memory and max_allocated_memory over-sized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NVTE_BATCH_MHA_P2P_COMM=1 # to force TransformerEngine to use batched send/recv for CP
export NCCL_DEBUG=WARN

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
export NNODES=4
export GPUS_PER_NODE=$ARNOLD_WORKER_GPU
export MASTER_ADDR=$METIS_WORKER_0_HOST
export MASTER_PORT=$port
export NODE_RANK=`expr $ARNOLD_ID - 0`

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# export NCCL_NVLS_ENABLE=1
export GLOO_SOCKET_IFNAME=eth0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export MODEL_NAME=$1
export DATASET=$2
export LAYER_NUM=$3
export HIDDEN_SIZE=$4
export NUM_ATTENTION_HEADS=$5
export FFN_HIDDEN_SIZE=$6
export EXPERT_NUM=$7
export TOPK=$8
export AUX_LOSS=$9
export BATCH_SIZE=${10}
export SEQ_LENGTH=${11}
export EXPERT_PARALLEL=${12}
export TENSOR_PARALLEL=${13}
export SAVE_OR_LOAD=${14}
export ITER=${15}

echo "MODEL_NAME: $MODEL_NAME"
echo "DATASET: $DATASET"
echo "LAYER_NUM: $LAYER_NUM"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "NUM_ATTENTION_HEADS: $NUM_ATTENTION_HEADS"
echo "FFN_HIDDEN_SIZE: $FFN_HIDDEN_SIZE"
echo "EXPERT_NUM: $EXPERT_NUM"
echo "TOPK: $TOPK"
echo "AUX_LOSS: $AUX_LOSS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "SEQ_LENGTH: $SEQ_LENGTH"
echo "EXPERT_PARALLEL: $EXPERT_PARALLEL"
echo "TENSOR_PARALLEL: $TENSOR_PARALLEL"
echo "SAVE_OR_LOAD: $SAVE_OR_LOAD"
echo "ITER: $ITER"

CHECKPOINT_PATH=$BASE_DIR/checkpoints/megatron_convergence/$MODEL_NAME
TOKENIZER_MODEL=$BASE_DIR/tokenizers/mixtral
DATA_PATH=$BASE_DIR/datasets/processed/$DATASET/mixtral_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $SEQ_LENGTH
    --num-layers $LAYER_NUM
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --make-vocab-size-divisible-by 1
)

if [[ "$MODEL_NAME" == qwen* ]]; then
    MODEL_ARGS+=(--add-qkv-bias)
    MODEL_ARGS+=(--norm-epsilon 1e-6)
fi

MOE_ARGS=(
    --num-experts $EXPERT_NUM
    --moe-router-topk $TOPK
    --moe-aux-loss-coeff $AUX_LOSS
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size $BATCH_SIZE
    --train-iters $ITER
    --lr 1.25e-6
    --lr-decay-style cosine
    --min-lr 1.25e-7
    --lr-warmup-fraction 0.1
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1.0e-5
    --init-method-std 0.01
    --clip-grad 0.0
    --bf16
    --no-create-attention-mask-in-dataloader
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size $EXPERT_PARALLEL
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 1 \
)

if [ $SAVE_OR_LOAD -eq 1 ]; then
    TRAINING_ARGS+=(--no-save-optim)
    TRAINING_ARGS+=(--ckpt-format torch)
    TRAINING_ARGS+=(--save $CHECKPOINT_PATH)
fi

if [ $SAVE_OR_LOAD -eq 2 ]; then
    TRAINING_ARGS+=(--load $CHECKPOINT_PATH)
    TRAINING_ARGS+=(--no-load-optim)
    TRAINING_ARGS+=(--no-load-rng)
fi

if [ $SAVE_OR_LOAD -eq 1 ]; then
    torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${LOGGING_ARGS[@]}
elif [ $SAVE_OR_LOAD -eq 2 ]; then
    torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${LOGGING_ARGS[@]} | tee training_log/megatron_convergence_${MODEL_NAME}_${DATASET}_batch${BATCH_SIZE}_seq${SEQ_LENGTH}_aux${AUX_LOSS}_${ARNOLD_ID}.log
fi
