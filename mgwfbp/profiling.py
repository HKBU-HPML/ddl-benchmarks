from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn.functional as F


class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self._parameter_names = {v: k for k, v
                                in model.named_parameters()}
        self._seq_keys = [k for k, v in model.named_parameters()]
        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._grad_accs = []
        self._handles = {}
        self.hook_done = False
        self._start = time.time()
        self._register_hooks()
        self._is_profiling = False

    def _register_hooks(self):
        for name, p in self.model.named_parameters():
            #p_tmp = p.expand_as(p)
            #grad_acc = p_tmp.grad_fn.next_functions[0][0]
            #grad_acc.register_hook(self._make_hook(name, p))
            #self._grad_accs.append(grad_acc)
            p.register_hook(self._make_hook(name, p))

    def _make_hook(self, name, p):
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)
            if len(self._backward_seq_keys) != len(self._seq_keys):
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(p.numel())
            if name not in self._handles:
                self._handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            #print(self._start, ',', ct, ', diff: ', ct-self._start, name, ', size: ', p.data.numel())
            self._handles[name].append(ct - self._start)
        return hook

    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_layerwise_times(self):
        num_trials = len(self._handles[self._seq_keys[0]])
        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
            layerwise_times_multipletest.append(layerwise_times)
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0)
        return layerwise_times, np.mean(totals)

    def _timestamp(self, name):
        return time.time()


def benchmark(model, fake_data, criterion, task='bert'):
    # Benchmark to achieve the backward time per layer
    p = Profiling(model)
    #data = torch.randn(input_shape)
    #target = torch.LongTensor(input_shape[0]).random_() % 1000
    if task == 'bert':
        input_ids, attention_masks, token_type_ids, position_ids, next_sentence_label, masked_lm_labels = fake_data
    else:
        inputs, labels = fake_data

    # Warmup
    #input_shape, output_shape = trainer.get_data_shape()
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50

    for i in range(iteration+warmup):
        if task == 'bert':
            prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)
            loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_label)
        else:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
        torch.cuda.synchronize()
       
        if i >= warmup:
            p.start()
        loss.backward()
        torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    return seq_keys[::-1], layerwise_times[::-1], p.get_backward_key_sizes()[::-1]


class CommunicationProfiler(object):
    def __init__(self, comm_op, sync_op, sizes=None):
        self.comm_op = comm_op
        self.sync_op = sync_op
        self.sizes = sizes

    def benchmark(self, num_iters=100):
        if self.sizes is None:
            small_sizes = [8*1024*i for i in range(1, 64)] # 1K to 1M
            large_sizes = [] #[1024*1024*i for i in range(8)] # 1M to 512M
            sizes = small_sizes+large_sizes
        else:
            sizes = self.sizes
        warmup = 5
        size = 1024
        tensor = torch.rand(size).float().cuda()
        stime = time.time()
        for i in range(warmup):
            name = 'warmup-%d' % i
            h = self.comm_op(tensor, average=True, name=name)
            self.sync_op(h)
        etime = time.time()
        elapsed_times = []
        for s in sizes:
            tensor = torch.rand(s).float().cuda()
            torch.cuda.synchronize()
            stime = time.time()
            for i in range(num_iters):
                name = 'run-size%d-%d'% (s, i)
                h = self.comm_op(tensor, average=True, name=name)
                self.sync_op(h)
            etime = time.time()
            elapsed_times.append((etime-stime)/num_iters)
        return sizes, elapsed_times


