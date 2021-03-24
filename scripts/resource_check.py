# job_submission.py COPYRIGHT Fujitsu Limited 2021 and FUJITSU LABORATORIES LTD. 2021
# Authors: Atsushi Kajita (kajita@fixstars.com), G R Ramdas Pillai (ramdas@fixstars.com)

import psutil
import json
import nvgpu
import time
import pynvml as nv
from multiprocessing import Process
from datetime import datetime
import subprocess
import os
import signal
from statistics import stdev

def cumulative_stats(p):
    mem = p.memory_full_info().uss
    cpup = p.cpu_percent(interval=0.1)
    return round(cpup, 4), round(mem/1e6, 4)

def device_status(device_index):
    handle = nv.nvmlDeviceGetHandleByIndex(device_index)
    device_name = nv.nvmlDeviceGetName(handle)
    device_name = device_name.decode('UTF-8')
    nv_procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
    utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    memory = nv.nvmlDeviceGetMemoryInfo(handle)
    clock_mhz = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
    temperature = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
    pids = []
    for nv_proc in nv_procs:
        pid = nv_proc.pid
        pids.append(pid)

    return {
        'type': device_name,
        'is_available': len(pids) == 0,
        'pids': ','.join([str(pid) for pid in pids]),
        'utilization': utilization,
        'clock_mhz': clock_mhz,
        'mem_total':  round(memory.total/1e6),
        'mem_used':  round(memory.used/1e6),
        'temperature': temperature,
        }

nv.nvmlInit()

resource_info = {}
tic = time.time()
with open('results/train.json', 'r') as f:
    train_file = json.load(f)

idxes = list(train_file.keys())

for idx in idxes:
    print("Processing", idx)
    tp = subprocess.Popen(['python','main.py','--host_filesystem','aws','--experiment_index',str(idx),'--run','train'], stdout=subprocess.DEVNULL)

    resource_info[idx] = {'cpup':[], 'mem':[], 'gpup':[], 'gpum':[]}
    ms = []
    for i in range(300):
        c, m = cumulative_stats(psutil.Process(tp.pid))
        resource_info[idx]['cpup'].append(c)
        resource_info[idx]['mem'].append(m)
        d = device_status(0)
        resource_info[idx]['gpup'].append(d['utilization'])
        resource_info[idx]['gpum'].append(d['mem_used'])
        ms.append(m)
        if(len(ms) > 5):
            ms.pop(0)
        if(len(ms) == 5):
            s = stdev(ms)
            if(s < 0.1):
                break
        time.sleep(1)

    os.kill(tp.pid, signal.SIGKILL)
    with open('resource_info.json', 'w') as f:
        json.dump(resource_info, f, indent=4)
    
nv.nvmlShutdown()    
toc = time.time()
print("Total time taken:", round(toc-tic, 2), "seconds")
