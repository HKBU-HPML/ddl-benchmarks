source ../configs/envs.conf
OSU_PATH=/home/esetstore/local/osu/libexec/osu-micro-benchmarks
LATENCY_BIN=$OSU_PATH/mpi/pt2pt/osu_latency
cmd=$LATENCY_BIN
nworkers=2
if [ "$rdma" = "0" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np 2 -hostfile cluster-cj$nworkers -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x LD_LIBRARY_PATH  \
    $cmd
elif [ "$rdma" = "1" ]; then
#100GbIB Config with RDMA 
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster-cj$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    $cmd
else
#100GbIB Config with Ethernet
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster-cj$nworkers -bind-to none -map-by slot \
    --mca pml ob1 --mca btl ^openib \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    $cmd
fi

