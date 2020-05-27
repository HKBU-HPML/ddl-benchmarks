workerid="${workerid:-0}"
bs="${bs:-32}"
senlen="${senlen:-64}"
#export BYTEPS_ENABLE_IPC=1
export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_WORKER_ID=$workerid
export DMLC_NUM_WORKER=4
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.31.229.66 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port

#RDMA
export DMLC_ENABLE_RDMA=1
export DMLC_INTERFACE=ib0
export DMLC_PS_ROOT_URI=10.149.160.57 # the scheduler IP
export DMLC_PS_ROOT_PORT=9000

#bpslaunch python3 imagenet_benchmark.py --model resnet50 --num-iters 100 --batch-size $bs
bpslaunch python3 bert_benchmark.py --sentence-len $senlen --num-iters 100 --batch-size $bs
