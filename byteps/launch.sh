#!/bin/bash
nworkers="${nworkers:-8}"
bs="${bs:-64}"
dnn="${dnn:-resnet50}"
senlen="${senlen:-64}"
rdma="${rdma:-0}"
debug="${debug:-1}"
threshold="${threshold:-0}"
nservers="${nservers:-0}"
bytescheduler="${bytescheduler:-0}"
source ../configs/envs.conf
CODEHOME=${CODEHOME}/byteps

if [ "$nservers" = "0" ]; then
    nservers=`expr $nworkers \/ 4` # we configure the number of servers be the same as WORKER
fi

if [ "$rdma" = "0" ]; then
    schedulerip=${ETH_SCHEDULER_IP}
    rmdaenv="DMLC_ENABLE_RDMA=0 " 
elif [ "$rdma" = "1" ]; then
    schedulerip=${IB_SCHEDULER_IP}
    rmdaenv="DMLC_ENABLE_RDMA=1 \
             DMLC_INTERFACE=${IB_INTERFACE} "
else
    schedulerip=${IB_SCHEDULER_IP}
    rmdaenv="DMLC_ENABLE_RDMA=0 \
             DMLC_INTERFACE=${IB_INTERFACE} "
fi

baseenv="DMLC_NUM_WORKER=$nservers \
         DMLC_NUM_SERVER=$nservers \
         DMLC_PS_ROOT_URI=$schedulerip \
         DMLC_PS_ROOT_PORT=1234 \
         BYTEPS_PARTITION_BYTES=4096000 \
         BYTEPS_SERVER_ENGINE_THREAD=8 \
         BYTEPS_SERVER_ENABLE_SCHEDULE=$bytescheduler \
         "

#if [ "$rdma" = "1" ]; then
baseenv=${baseenv}${rmdaenv}
#fi


if [ "$dnn" = "bert" ] || [ "$dnn" = "bert_base" ]; then
    cmd="$PY bert_benchmark.py --model $dnn --sentence-len $senlen --batch-size $bs"
else
    cmd="$PY imagenet_benchmark.py --model $dnn --batch-size $bs"
fi

servercmd=$baseenv"DMLC_ROLE=server \
          $LAUNCHBIN"

schedulercmd=$baseenv"DMLC_ROLE=scheduler \
          $LAUNCHBIN"

workerbasecmd=$baseenv"DMLC_ROLE=worker NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
          $LAUNCHBIN $cmd"

workerbasecmd="BYTEPS_FUSION_THRESHOLD=$threshold "$workerbasecmd
#if [ "$wfbp" = "0" ]; then
#    workerbasecmd="BYTEPS_FUSION_THRESHOLD=536870912 "$workerbasecmd
#else
#    workerbasecmd="BYTEPS_FUSION_THRESHOLD=0 "$workerbasecmd
#fi

# start scheduler
echo "============Scheduler ============="
echo "Scheduler"
echo "Host: "$schedulerip
echo "CMD: "$schedulercmd
if [ "$debug" = "0" ]; then
    ssh -o "StrictHostKeyChecking no" $schedulerip "$schedulercmd" &
fi
workerprefix=${WORKER_HOST_PREFIX}
# start parameter servers
echo "============PS ============="
for serverid in `seq 1 ${nservers}`
do
    serverid=`expr 8 \+ $serverid` # server start at gpu9 
    host=$workerprefix$serverid
    echo "Host: "$host
    echo "CMD: "$servercmd
    echo 
    if [ "$debug" = "0" ]; then
        ssh -o "StrictHostKeyChecking no" $host "$servercmd" &
    fi
done

# start workers 
echo "============Worker ============="
for workerid in `seq 1 ${nservers}`
do
    host=$workerprefix$workerid
    workerid0=`expr $workerid \- 1` 
    if [ "$rdma" = "0" ]; then
        if [ "$host" = "gpu7" ] || [ "$host" = "gpu8" ]; then
            workercmd="cd $CODEHOME; DMLC_INTERFACE=enp137s0f0 DMLC_WORKER_ID=$workerid0 $workerbasecmd"
        else
            workercmd="cd $CODEHOME; DMLC_INTERFACE=enp136s0f0 DMLC_WORKER_ID=$workerid0 $workerbasecmd"
        fi
    else
        workercmd="cd $CODEHOME; DMLC_WORKER_ID=$workerid0 $workerbasecmd"
    fi
    echo "Worker "$workerid0
    echo "Host: "$host
    echo "CMD: "$workercmd
    echo 
    if [ "$debug" = "0" ]; then
        if [ "$workerid" = "$nservers" ]; then
            ssh -o "StrictHostKeyChecking no" $host "$workercmd"
        else
            ssh -o "StrictHostKeyChecking no" $host "$workercmd" &
        fi
    fi
done
