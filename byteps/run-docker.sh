#docker run -v /home/shaohuais/repos/benchmarks:/benchmarks -it --net=host bytepsimage/pytorch bash 
#docker run -v /tmp/benchmarks:/benchmarks -it --net=host bytepsimage/pytorch bash  #TCP

#RDMA
docker run -v /tmp/benchmarks:/benchmarks -it --net=host --shm-size=32768m --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/pytorch bash

#docker run -v /tmp/benchmarks:/benchmarks -it --net=host --device /dev/infiniband/rdma_cm --device /dev/infiniband/issm0 --device /dev/infiniband/ucm0 --device /dev/infiniband/umad0 --device /dev/infiniband/uverbs0 --cap-add IPC_LOCK bytepsimage/tensorflow bash

