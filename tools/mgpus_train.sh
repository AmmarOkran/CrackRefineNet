export CUDA_VISIBLE_DEVICES=1

CONFIG=$1
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=2 \
    --master_port=29500 \
    tools/train.py $CONFIG  \
    --launcher pytorch
