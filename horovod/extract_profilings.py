import numpy as np

def extract(fn):
    with open(fn, 'r') as f:
        total_FLOPs = 0
        for line in f.readlines():
            if line.find('FP Instructions') > 0:
                items = line.split()
                invocations, avg = items[0], items[-1]
                total_FLOPs += float(invocations) * float(avg)
        print('fn: ', fn, ', GFLOPs: ', total_FLOPs/(1e9))


if __name__ == '__main__':
    #extract('resnet50.prof')
    extract('bert.prof')
