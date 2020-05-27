#!/bin/bash
cmd="cd /home/esetstore/repos/ddl-benchmarks/byteps; ./kill.sh"
for serverid in `seq 1 16`
do
    host=gpu$serverid
    #echo $host
    ssh -o "StrictHostKeyChecking no" $host $cmd &
done
