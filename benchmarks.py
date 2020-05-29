import os
import datetime
import subprocess
import os
import numpy as np
import time

DEBUG=0

LOGPREFIX='pcie'
a2amethods=['bspa2a', 'wfbp', 'mgwfbp', 'bytescheduler']
psmethods=['bspps','wfbpps', 'byteschedulerps']
methods=a2amethods+psmethods
tasks=[('resnet50', 64), ('bert', 8), ('bert_base', 64)]
nworkers=[4, 8, 16, 32] 
rdmas=[0, 1, 2]
NUM_OF_TRIES=5
exp_log='exp.log'

def configs():
    cfg = {}
    cfg['LOGPREFIX'] = LOGPREFIX
    cfg['methods'] = methods
    cfg['tasks'] = tasks
    cfg['nworkers'] = nworkers
    cfg['rdmas'] = rdmas
    cfg['NUM_OF_TRIES'] = NUM_OF_TRIES
    cfg['exp_log'] = exp_log
    return cfg

def gen_cmd(rdma, method, task, nworker):
    LOGHOME = os.path.abspath(os.getcwd())
    folder = method 
    compressor = 'none'
    threshold = 0
    ismgwfbp = 1
    if method in a2amethods:
        if method in ['signum', 'eftopk', 'bspa2a', 'wfbp', 'dgcsampling']:
            if method in ['signum', 'eftopk', 'dgcsampling']:
                compressor = method 
            folder = 'mgwfbp'
            ismgwfbp = 0
            if method == 'wfbp':
                threshold = 0 
            else:
                threshold = 536870912
        logfile = '%s/logs/%s/rdma-%d-method-%s-dnn-%s-bs-%d-nworker-%d-compressor-%s-thres-%d.log' % (LOGHOME, LOGPREFIX, rdma, method, task[0], task[1], nworker, compressor, threshold)
        cmd = 'cd %s;'% folder
        cmd += 'rdma=%d dnn=%s bs=%d nworkers=%d compressor=%s threshold=%d mgwfbp=%d ./horovod_mpi_cj.sh >> %s 2>&1' % (rdma, task[0], task[1], nworker, compressor, threshold, ismgwfbp, logfile)
    else: #PS
        threshold=0
        bytescheduler=0
        if method == 'bspps':
            threshold=536870912
        if method == 'byteschedulerps':
            bytescheduler=1
        logfile = '%s/logs/%s/rdma-%d-method-%s-dnn-%s-bs-%d-nworker-%d-compressor-%s-thres-%d.log' % (LOGHOME, LOGPREFIX, rdma, method, task[0], task[1], nworker, compressor, threshold)
        folder='byteps'
        cmd = 'cd %s;'% folder
        cmd += 'debug=0 rdma=%d dnn=%s bs=%d nworkers=%d compressor=%s threshold=%d mgwfbp=%d bytescheduler=%d ./launch.sh >> %s 2>&1' % (rdma, task[0], task[1], nworker, compressor, threshold, ismgwfbp, bytescheduler, logfile)
    return cmd, logfile

def check_if_finished(cmd):
    if not os.path.isfile(exp_log):
        with open(exp_log, 'w+') as f:
            f.write('')
            return False
    with open(exp_log, 'r') as f:
        for line in f.readlines():
            if line.find(cmd) >= 0:
                return True
        return False

def flag_as_finished(cmd):
    with open(exp_log, 'a+') as f:
        f.write(cmd+'\n')

def execute(cmd, logfile):
    print('%s' % cmd)
    if DEBUG:
        return 0,0
    finished = check_if_finished(cmd)
    if not finished:
        with open(logfile, 'w+') as f:
            x = datetime.datetime.now()
            f.write('#Date: %s\n#CMD: %s\n' % (x.strftime("%b %d %Y %H:%M:%S"), cmd))
        for i in range(NUM_OF_TRIES):
            try:
                subprocess.check_output(cmd, shell=True)
            except Exception as e:
                print('cmd: %s ERROR: %s' % (cmd, e))
        flag_as_finished(cmd)
    speed = extract_log(logfile)
    return speed

def extract_log(logfile):
    with open(logfile) as f:
        speeds = []
        for line in f.readlines():
            if line.find('Total') >= 0:
                speed = float(line.split(': ')[-1].split()[0])
                speeds.append(speed)
        mean = np.mean(speeds)
        std = np.std(speeds)
        return mean, std
    return 0, 0

def init_reports():
    reports = {}
    for rdma in rdmas:
        reports[rdma] = {}
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            reports[rdma][task_str] = {}
            for method in methods:
                reports[rdma][task_str][method] = []
    return reports

def write_reports(reports):
    import json
    print('==== All Reports ======')
    for rdma in rdmas:
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            for method in methods:
                print('rdma:%d,%s,%s'%(rdma, method, ','.join(reports[rdma][task_str][method])))
    with open('reports.json', 'w') as fp:
        json.dump(reports, fp)

def main():
    reports = init_reports()
    for rdma in rdmas:
        for task in tasks:
            task_str = '%s_%d' % (task[0], task[1])
            for method in methods:
                for nworker in nworkers:
                    cmd, logfile = gen_cmd(rdma, method, task, nworker)
                    speed = execute(cmd, logfile)
                    #speed_str = '%.3f+-%.3f' % speed
                    speed_str = '%.3f' % speed[0]
                    reports[rdma][task_str][method].append(speed_str)
                    print('Speed: ', speed)
                    print
                    if method in psmethods:
                        killcmd='cd byteps;./stop.sh'
                        subprocess.check_output(killcmd, shell=True)
                        time.sleep(1)
                print('rdma:%d,%s,%s'%(rdma, method, ','.join(reports[rdma][task_str][method])))
    write_reports(reports)


if __name__ == '__main__':
    main()
