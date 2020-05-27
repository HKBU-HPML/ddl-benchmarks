/usr/local/cuda/bin/nvprof --log-file bert.prof --metrics inst_fp_32 python bert_benchmark.py --batch-size 8 --num-warmup-batches 0 --num-iters 1
/usr/local/cuda/bin/nvprof --log-file resnet50.prof --metrics inst_fp_32 python imagenet_benchmark.py --batch-size 64 --num-warmup-batches 0 --num-iters 1
