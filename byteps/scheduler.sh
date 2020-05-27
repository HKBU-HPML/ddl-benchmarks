# now you are in docker environment
#export BYTEPS_ENABLE_IPC=1
export DMLC_NUM_WORKER=4
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.31.229.66 # the scheduler IP
export DMLC_PS_ROOT_PORT=1234  # the scheduler port

#RDMA
export DMLC_ENABLE_RDMA=1
export DMLC_INTERFACE=ib0
export DMLC_PS_ROOT_URI=10.149.160.57 # the scheduler IP
export DMLC_PS_ROOT_PORT=9000

bpslaunch
