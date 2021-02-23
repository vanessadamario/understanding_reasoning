import ray
import sys
import argparse
import re
import json
import math
import os
from pathlib import Path
from datetime import datetime
import warnings
import time
from prettytable import PrettyTable
from reprint import output
import signal

@ray.remote(num_gpus=0.3)
def act(task, idx, work_dir):
    import subprocess
    import os
    import numpy as np
    from errno import errorcode
    import socket
    import glob

    hostname = str(socket.gethostname())

    # import main
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    tag = 'expIdx' + "_" + str(idx)
    
    prefix = work_dir + '/logs/'
    
    log_path = prefix+tag
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    else:
        files = glob.glob(log_path + '/*')
        for f in files:
            os.remove(f)

    log_redirection = " 2>&1 > " +  log_path + "/log.txt"
    path_movement = 'cd ' + work_dir + '; '

    ret = os.system(path_movement + '/bin/bash -c "set -o pipefail; ' + task + log_redirection + '"')

    if(ret != 0):
        py_error = int(bin(ret)[2:-8], 2)
        ret = py_error
        if(py_error != 0):
            print("Error: Received errno string:", errorcode[py_error])
        else:
            os_error = int(bin(ret)[-8:], 2)
            ret = os_error
            print("Error: Received errno string:", errorcode[os_error])
    return ret


    
def get_multi_commands(command, exps):

    multi_cmds = []
    command = ' '.join(command)
    for i in exps:
        multi_cmds.append(command + ' --experiment_index ' + str(i))

    return multi_cmds


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_size(size_str):
    size_str = size_str.upper()
    parsed = re.split('([\d.]+)', size_str)

    for i in parsed:
        if(is_number(i) == True):
            number = float(i)
        if(i.strip().isalpha() == True):
            unit = i.strip()
    if(unit not in units.keys()):
        raise ValueError("Unknown size unit provided!")
    return int(float(number)*units[unit])

def get_size(size_str, unit='m'):
    if(unit == 'm'):
        return max(MINIMUM_SIZE, int(math.ceil(parse_size(size_str)/1e6)))
    if(unit == 'g'):
        return max(MINIMUM_SIZE, int(math.ceil(parse_size(size_str)/1e9)))  

def getargs(argv):
    
    backup_args = sys.argv[1:]
    allargs = argv

    raise_error = False

    try:
        cmdpos = allargs.index('--')
    except:
        cmdpos = len(allargs)
        raise_error = True

    if(cmdpos == (len(allargs) - 1)):
        raise_error = True

    command = allargs[cmdpos+1:]

    sys.argv[1:] = allargs[:cmdpos]
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_index', type=str, required=True)
    parser.add_argument('--work_path', type=str, required=True)
    
    args = parser.parse_args()
    exp_ids = args.experiment_index.split(',')
    exps = []
    for i in exp_ids:
        if('-' in i):
            j = [int(x) for x in i.split('-')]
            for k in range(j[0], j[1]+1):
                exps.append(k)
        else:
            exps.append(int(i))

    if(raise_error == True):
        raise ValueError("No command entered")

    sys.argv[1:] = backup_args

    return exps, command, args.work_path

def submit_job(multi_cmds, exps, wrk_path):
    RUNS=len(exps)
    print("Submitting job to ray")
    rets = [act.remote(multi_cmds[i], exps[i], wrk_path) for i in range(len(exps))]
    return rets

def mainrun(argv):

    # Parse the command line arguements
    exps, command, wrk_path = getargs(argv)

    multi_cmds = get_multi_commands(command, exps)

    # Initialize the ray
    try:
        ray.init(address='auto', _redis_password='5241590000000000')
    except:
        raise RuntimeError("Ray not started")


    rets = submit_job(multi_cmds, exps, wrk_path)
    starttime = datetime.now()
    et = {}
    for ctr in range(len(rets)):
        et[ctr] = datetime.now() - starttime
    with output(initial_len=6 + len(rets), interval=0) as output_lines:
        while True:
            rd = []
            nrd = []
            for ctr in range(len(rets)):
                obj = rets[ctr]
                r, n = ray.wait([obj], timeout=0)
                for i in r:
                    rd.append(i)
                for i in n:
                    nrd.append(i)
                    et[ctr] = datetime.now() - starttime
            
            case_info = os.path.basename(os.getcwd())
            dash_data = {}
            dash_data['last_updated'] = str(datetime.now().strftime("%Y.%m.%d - %H:%M:%S%p"))
            dash_data['current_case'] = case_info
            dash_data['info'] = {}
            dash_data['info']['Tag'] = []
            dash_data['info']['Status'] = []
            dash_data['info']['Elapsed Time'] = []

            x = PrettyTable()
            x.title = 'Job Status - ' + str(datetime.now().strftime("%Y.%m.%d - %H:%M:%S%p"))
            x.field_names = ['Tag', 'Status', 'Elapsed Time']
            for i in range(len(rets)):
                obj = rets[i]
                tagname = 'expIdx' + '_' + str(exps[i])
                if(obj in rd):
                    dash_data['info']['Tag'].append(tagname)
                    dash_data['info']['Status'].append('Finished')
                    dash_data['info']['Elapsed Time'].append(str(et[i]))
                    x.add_row([tagname, "Finished", str(et[i])])
                if(obj in nrd):
                    dash_data['info']['Tag'].append(tagname)
                    dash_data['info']['Status'].append('In Progress...')
                    dash_data['info']['Elapsed Time'].append(str(et[i]))
                    x.add_row([tagname, "In Progress...", str(et[i])])

            with open(wrk_path + '/progress.json', 'w') as f:
                json.dump(dash_data, f, indent=4)

            xs = str(x).split('\n')
            for i in range(len(xs)):
                output_lines[i] = xs[i]

            if(len(rd) == len(exps)):
                break
            time.sleep(5)

    print("Finished all tasks. Exiting.")



if __name__ == '__main__':
    mainrun(sys.argv[1:])

