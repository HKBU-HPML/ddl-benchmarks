import os
import datetime
import subprocess
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('fivethirtyeight') 

from benchmarks import *

HATCH = ['//', '--', '\\\\', 'xxx', '//', 'ooo', '**', 'OO', '']
COLOR = ['#70ad47',
'#A9D18E',
#'#4672C4',
#'#3F5EBA',
#'#70ad47',
'#F1B111',
'#F1B183',
'#c55a11',
'#F55a11',
'white',
]
#COLOR = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']

METHOD_NAMES = {
        'bspa2a': 'BSP-A2A',
        'bspps': 'BSP-PS',
        'signum': 'Signum',
        'eftopk': 'TopK-SGD', 
        'wfbpps': 'WFBP-PS',
        'wfbp': 'WFBP-A2A',
        'mgwfbp': 'MG-WFBP',
        'byteschedulerps': 'ByteScheduler-PS',
        'bytescheduler': 'ByteScheduler-A2A',
        'optimal': 'Optimal'
        }

METHOD_HACHES = {
        'bspps': '///',
        'bspa2a': '..',
        'signum': '\\\\\\',
        'eftopk': 'xx', 
        'wfbpps': '\\\\\\',
        'wfbp': 'oo',
        'mgwfbp': '**',
        'byteschedulerps': '++',
        'bytescheduler': 'OO',
        'optimal': ''
        }

METHOD_COLORS = {
        'bspa2a': '#F55a11', 
        'bspps':  '#F1B111', 
        'signum': '#F55a11', 
        'eftopk':  '#F55a11',
        'wfbpps':  '#F1B111',
        'wfbp':  '#F55a11',
        'mgwfbp':  '#F55a11',
        'byteschedulerps': '#F1B111',
        'bytescheduler': '#F55a11',
        'optimal': 'white'
        }

#METHOD_COLORS = {
#        'bspa2a': '#A9D18E', 
#        'bspps':  '#4672C4', 
#        'signum': '#3F5EBA', 
#        'eftopk':  '#70ad47',
#        'wfbpps':  '#F1B111',
#        'wfbp':  '#F1B183',
#        'mgwfbp':  '#c55a11',
#        'byteschedulerps': '#B55a11',
#        'bytescheduler': '#F55a11',
#        'optimal': 'white'
#        }


OUTPUT_PATH = '/media/tmp/ieeenetwork'

single_throughputs = {
        'resnet50_64': 253.1,
        'bert_8': 49,
        'bert_base_64': 212.8
        }
candidate_texts = {}

def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def record_rects(rects, task_str):
    single_throughput = single_throughputs[task_str]
    for i, rect in enumerate(rects):
        height = rect.get_height()
        value = height / (single_throughput)
        if i in candidate_texts:
            current_value = candidate_texts[i][0]
            if value > current_value:
                candidate_texts[i] = (value, rect)
        else:
            candidate_texts[i] = (value, rect)

def autolabel(multi_rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for k in multi_rects:
        rect = multi_rects[k][1]
        value = multi_rects[k][0] 
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+100.0,
                '%.1fx' % value,
                ha='center', va='bottom', rotation=90)


def read_reports():
    reports = init_reports()
    for rdma in rdmas:
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            for method in methods:
                for nworker in nworkers:
                    cmd, logfile = gen_cmd(rdma, method, task, nworker)
                    try:
                        speed = extract_log(logfile)
                        #speed_str = '%.3f+-%.3f' % speed
                    except:
                        speed_str = (0, 0)
                    reports[rdma][task_str][method].append(speed)
                #print('rdma:%d,%s,%s,%s'%(rdma, task_str, method, ','.join(['%.3f+-%.3f'% speed for speed in reports[rdma][task_str][method]])))
                print('rdma:%d,%s,%s,%s'%(rdma, task_str, method, ','.join(['%.3f'% speed[0] for speed in reports[rdma][task_str][method]])))
            reports[rdma][task_str]['optimal'] = [(single_throughputs[task_str]*nworker, 0) for nworker in nworkers]
    return reports

def compare_rdma():
    reports = read_reports()
    rdmas = [0, 2, 1] # 10GbE, 100GbE, 100GbIB
    tasks = list(reports[0].keys())
    methods=['bspps', 'bspa2a', 'wfbpps', 'wfbp', 'mgwfbp', 'byteschedulerps', 'bytescheduler']
    for method in methods:
        for t in tasks:
            data1 = reports[rdmas[0]][t]
            ngroups = len(data1[methods[0]]) - 1
            values1 = [mean for mean, std in data1[method]][:ngroups]

            data2 = reports[rdmas[1]][t]
            values2 = [mean for mean, std in data2[method]][:ngroups]

            data3 = reports[rdmas[2]][t]
            values3 = [mean for mean, std in data3[method]][:ngroups]
            speeds_100GbE_vs_10GbE = np.array(values2)/np.array(values1)
            speeds_100GbIB_vs_100GbE = np.array(values3)/np.array(values2)
            print('-->method: ', method, ', task: ', t)
            print(speeds_100GbE_vs_10GbE)
            print(speeds_100GbIB_vs_100GbE)
            print

def plots():
    fig, ax = plt.subplots()

    reports = read_reports()
    print
    rdma = 0
    tasks = list(reports[rdma].keys())
    task = tasks[0]
    data = reports[rdma][task]
    #methods = list(data.keys())
    #methods=['bspps', 'bspa2a', 'signum', 'eftopk', 'wfbpps', 'wfbp', 'mgwfbp', 'bytescheduler', 'optimal']
    methods=['bspps', 'bspa2a', 'wfbpps', 'wfbp', 'mgwfbp', 'byteschedulerps', 'bytescheduler'] #, 'optimal']
    print('methods: ', methods)
    ngroups = 4 #len(data[methods[0]])
    bar_width = 0.9/len(methods) 
    ind = np.arange(ngroups)

    bars = []
    for i, method in enumerate(methods):
        values = [mean for mean, std in data[method]][:ngroups]
        errs = [std for mean, std in data[method]][:ngroups]
        color = METHOD_COLORS[method]
        hatch = METHOD_HACHES[method]
        bar = ax.bar(ind-bar_width*len(methods)/2, values, bar_width, yerr=errs, color=color, edgecolor='black', hatch=hatch, ecolor='black', capsize=2)
        record_rects(bar, task)
        #bar = ax.bar(ind, values, bar_width, edgecolor='black', hatch=HATCH[i%len(methods)])
        ind = ind + bar_width
        bars.append(bar)
    autolabel(candidate_texts, ax)
    nmethods = len(methods)
    xticks = np.arange(ngroups-1 + bar_width*(nmethods / 2.0))
    print('xticks: ', xticks)
    #ax.grid(True)
    ax.grid(linestyle=':')
    ax.set_xticks(xticks)
    xlabels = tuple([2**(2+g) for g in range(ngroups)])
    ax.set_xticklabels(xlabels, size='x-large')
    ax.legend(tuple([bar[0] for bar in bars]), tuple([METHOD_NAMES[m] for m in methods]), loc=2, ncol=1, fontsize='x-large')
    ax.set_xlabel('# of GPUs')
    ax.set_ylabel('Throughput (Samples per second)')
    print('task: ', task)
    if task.find('bert_8') >=0:
        ax.set_ylim(top=450)
    elif task.find('bert_base_64') >=0:
        ax.set_ylim(top=5800)
    else:
        ax.set_ylim(top=9000)
    update_fontsize(ax, 14)
    #plt.show()
    plt.savefig(os.path.join(OUTPUT_PATH, '%s-rdma%d.pdf'%(task,rdma)), bbox_inches='tight')

if __name__ == '__main__':
    plots()
    #compare_rdma()
