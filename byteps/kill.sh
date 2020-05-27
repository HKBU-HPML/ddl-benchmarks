kill -9 `ps aux|grep "python bert_benchmark.py" |awk '{print $2}'` 2> null
kill -9 `ps aux|grep "python imagenet_benchmark.py" |awk '{print $2}'` 2> null
kill -9 `ps aux|grep "bpslaunch" |grep -v grep |awk '{print $2}'` 2> null
