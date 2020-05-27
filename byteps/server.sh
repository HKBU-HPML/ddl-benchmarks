# now you are in docker environment
#export BYTEPS_ENABLE_IPC=1
export DMLC_NUM_WORKER=4
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.31.229.66 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234  # the scheduler port
export BYTEPS_PARTITION_BYTES=4096000
export BYTEPS_SERVER_ENGINE_THREAD=8
export BYTEPS_SERVER_ENABLE_SCHEDULE=0

#RDMA
export DMLC_ENABLE_RDMA=1
export DMLC_INTERFACE=ib0
export DMLC_PS_ROOT_URI=10.149.160.57 # the scheduler IP
export DMLC_PS_ROOT_PORT=9000

bpslaunch
