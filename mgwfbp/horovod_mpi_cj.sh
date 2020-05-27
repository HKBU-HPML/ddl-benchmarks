#!/bin/bash
nworkers="${nworkers:-8}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
compressor="${compressor:-none}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
mgwfbp="${mgwfbp:-1}"
threshold="${threshold:-0}"
source ../configs/envs.conf

if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    benchfile="bert_benchmark.py --model $dnn --sentence-len $senlen"
else
    benchfile="imagenet_benchmark.py --model $dnn"
fi

if [ "$compressor" = "none" ]; then 
    if [ "$mgwfbp" = "1" ]; then 
        cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --mgwfbp"
    else
        cmd="$PY $benchfile --density 1 --compressor $compressor --batch-size $bs --threshold $threshold"
    fi
else
    cmd="$PY $benchfile --density 0.001 --compressor $compressor --batch-size $bs --threshold 67108864"
fi
echo $cmd

#10GbE Config
if [ "$rdma" = "0" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ${ETH_MPI_BTC_TCP_IF_INCLUDE} \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=${ETH_INTERFACE} \
    -x NCCL_IB_DISABLE=1 \
    $cmd
elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=INFO \
    $cmd
else
#100GbIB Config with Ethernet
cmd="$cmd --rdma"
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ../configs/cluster$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ${IB_INTERFACE} \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=${IB_INTERFACE} \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_NET_GDR_LEVEL=0 \
    -x NCCL_NET_GDR_READ=0 \
    -x NCCL_IB_CUDA_SUPPORT=0 \
    $cmd
fi

