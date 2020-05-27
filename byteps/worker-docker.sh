#nvidia-docker run -v /tmp/benchmarks:/benchmarks -it --net=host --shm-size=32768m bytepsimage/pytorch bash

#RDMA
nvidia-docker run -v /tmp/benchmarks:/benchmarks -it --net=host --shm-size=32768m --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/pytorch bash

