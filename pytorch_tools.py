"""
Autor: Alexander Schechtel
FH-Aachen FB5 Informatik.

"""

print("Tools 1.0 | Setze Umgebungsvariablen")

### DDP Standart 
PYTORCH_DIST_BACKEND   = 'nccl'
PYTORCH_DIST_DDP_PORT  = '23456'
NCCL_SOCKET_NTHREADS   = '4'
NCCL_NSOCKS_PERTHREAD  = '2'
PYTORCH_MODULE_LOG     = "True"
### PyTorch RPC-RRef Model-Parallel
PYTORCH_MODEL_PARALLEL__GLOO_SOCKET_IFNAME = "eno1"
PYTORCH_MODEL_PARALLEL__NCCL_SOCKET_IFNAME = "eno1"
PYTORCH_MODEL_PARALLEL__TP_SOCKET_IFNAME   = "eno1"
PYTORCH_MODEL_PARALLEL__START_ON_RANK      = "0"

import os
os.environ['PYTORCH_DIST_BACKEND']  = PYTORCH_DIST_BACKEND
os.environ['PYTORCH_DIST_DDP_PORT'] = PYTORCH_DIST_DDP_PORT
os.environ['NCCL_SOCKET_NTHREADS']  = NCCL_SOCKET_NTHREADS
os.environ['NCCL_NSOCKS_PERTHREAD'] = NCCL_NSOCKS_PERTHREAD
os.environ['PYTORCH_MODULE_LOG']    = PYTORCH_MODULE_LOG
### PyTorch RPC-RRef Model-Parallel
os.environ['GLOO_SOCKET_IFNAME'] = PYTORCH_MODEL_PARALLEL__GLOO_SOCKET_IFNAME
os.environ['NCCL_SOCKET_IFNAME'] = PYTORCH_MODEL_PARALLEL__NCCL_SOCKET_IFNAME
os.environ['TP_SOCKET_IFNAME']   = PYTORCH_MODEL_PARALLEL__TP_SOCKET_IFNAME
os.environ['PYTORCH_MODEL_PARALLEL__START_ON_RANK']      = PYTORCH_MODEL_PARALLEL__START_ON_RANK


  
if bool(PYTORCH_MODULE_LOG):
    print(f"PYTORCH_DIST_BACKEND: {PYTORCH_DIST_BACKEND}\nPYTORCH_DIST_DDP_PORT: {PYTORCH_DIST_DDP_PORT}\nNCCL_SOCKET_NTHREADS: {NCCL_SOCKET_NTHREADS}\nNCCL_NSOCKS_PERTHREAD: {NCCL_NSOCKS_PERTHREAD}\nPYTORCH_MODULE_LOG: {PYTORCH_MODULE_LOG}")
    print(f"\nRPC Config:\nGLOO_SOCKET_IFNAME: {PYTORCH_MODEL_PARALLEL__GLOO_SOCKET_IFNAME}, NCCL_SOCKET_IFNAME: {PYTORCH_MODEL_PARALLEL__NCCL_SOCKET_IFNAME}\nTP_SOCKET_IFNAME: {PYTORCH_MODEL_PARALLEL__TP_SOCKET_IFNAME}\
    , START_ON_RANK: {PYTORCH_MODEL_PARALLEL__START_ON_RANK}")

    
## Imports
#############################################################################################

## DDP, Hooks, Optimierer
from torch.nn.parallel import DistributedDataParallel as _DDP

from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as _sgd
import torch.distributed.algorithms.model_averaging.averagers as _averagers
from torch.distributed.optim import PostLocalSGDOptimizer
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
  PostLocalSGDState,
  post_localSGD_hook,
 )
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_wrapper

## Dask
from dask.distributed import Client as _Client
from dask.distributed import  wait
## Unsere Module 
import pytorch_dispatcher_resulthandler
## Sonstiges
import time
import datetime
import torch
from dask_cuda import LocalCUDACluster
from typing import Dict, List
    
_Dask_Client = None 



############# PyTorch Hooks 
#########################################################################################################################################################


############# PowerSGD 
def register_DDP_PowerSGD_hook(model, device_id, matrix_approximation_rank:int=12, start_powerSGD_iter:int=2):
    """Nutze den PowerSGD Hook, um die DDP Kommunikation bei großen Modellen zu verbessern.
    - model: Das Model
    - device_id: device_id  ->  worker_rank % torch.cuda.device_count()
    - matrix_approximation_rank:int=12: Kompressionsrate. 
    - start_powerSGD_iter:int=2: Iterationen bis All Reduce 
    """
    ## Hook start ##  
    if not isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel): 
        model = _DDP(model, device_ids=[device_id])
                            
    state = _sgd.PowerSGDState(
          process_group             = model.process_group,                        # Model Prozessgruppe
          matrix_approximation_rank = matrix_approximation_rank,                  
          start_powerSGD_iter       = start_powerSGD_iter                         
            )  
    model.register_comm_hook(state, _sgd.powerSGD_hook)
    return model
    ## Hook End ## 
    
    
############### Batched PowerSGD

def register_DDP_batched_PowerSGD_hook(model, device_id, matrix_approximation_rank:int=1, start_powerSGD_iter:int=16):
    """Noch nicht verfügbar
    """
    pass


## bf16 ##    
def register_hook_bf16_compress_hook():
    """Noch nicht verfügbar.
    """
    # Warning: This API is experimental, and it requires NCCL version later than 2.9.6.
    pass

############# PowerSGD fp16 ##
def register_fp16_compress_wrapper_SgdPower_hook(model, device_id, matrix_approximation_rank:int=12, start_powerSGD_iter:int=2):
    """Nutze den PowerSGD Hook mit fp16 wrapper, um die DDP Kommunikation bei großen Modellen zu verbessern.
    - model: Das Model
    - device_id: device_id  ->  worker_rank % torch.cuda.device_count()
    - matrix_approximation_rank:int=12: Kompressionsrate. 
    - start_powerSGD_iter:int=2: Iterationen bis All Reduce 
    """
 
    if not isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel): 
        model = _DDP(model, device_ids=[device_id])
                            
    state = _sgd.PowerSGDState(
          process_group             = model.process_group,                        # Model Prozessgruppe
          matrix_approximation_rank = matrix_approximation_rank,                  
          start_powerSGD_iter       = start_powerSGD_iter,
          batch_tensors_with_same_shape = True  # Added
            )
    
    model.register_comm_hook(state, fp16_compress_wrapper(_sgd.powerSGD_hook))
    
    return model
     
    
## Distributed Optimizer: PostLocalSGDOptimize 
########################################################
def register_distOpt_PostLocalSGDOpt(model,  device_id, local_optim, period:int=4, warm_up:int=100):
    """Nutzen den PostLocalSGD Optimierer.
    - Model: das Model.
    - device_id: device_id  ->  worker_rank % torch.cuda.device_count()
    - local_optim: lokaler Optimierer 
    - period:int=4: Der PostLocalSGDOpt mittel Global alle 4 Schritte.
    - warm_up:int=100: Lokale Gradientenmittlung in den ersten 100 Schritte.
    """
    
    if not isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel): 
        model = _DDP(model, device_ids=[device_id])
        
    # Register a post-localSGD communication hook.
    state = PostLocalSGDState(process_group=model.process_group, subgroup=None, start_localSGD_iter=warm_up)
    model.register_comm_hook(state, post_localSGD_hook)
    
    opt = PostLocalSGDOptimizer(
        optim=local_optim,
        averager=_averagers.PeriodicModelAverager(period=period, warmup_steps=warm_up)
    )
    return model, opt 


## Distributed Optimizer: ZeroRedundancyOptimizer 
########################################################
def register_distOpt_ZeroRedundancyOptimizer(model, local_optim_class:torch.optim, **kwargs):
    """ Noch nicht verfügbar
    """
    pass


    


## Start Training
VALID_RUN_STRINGS={"ddp", "model_parallel", "parameter_server"}
################################################################################################################
def run(train:callable, client,  *args, date_time_hour_offset:int=0, pytorch_mode:str="ddp", **kwargs):  # rpc_model_parallel_mode:bool=False
    """Starte das Training.
    train: Trainingsfunktion
    client: Dask Client
    date_time_hour_offset: Rechne +/- zur ausgegebenen Systemzeit
    **kwargs: Übergebe der Trainingsfunktion Parameter, wie: a=100,  data={...}
    """
    if pytorch_mode not in VALID_RUN_STRINGS:
        raise ValueError("run(): pytorch_mode must be one of %r." % VALID_RUN_STRINGS)
        
    workers = client.has_what().keys()
    n_workers = len(workers)
    print(f"Worker count: {n_workers}")
    
    #print(f"args: {args}, kwargs: {kwargs}")

    client.wait_for_workers(n_workers)

    rh = pytorch_dispatcher_resulthandler.DaskResultHandler("my_channel")   
    
    #########
    
    futures = pytorch_dispatcher_resulthandler.run(client, train, pytorch_mode, **kwargs) 
    dask_futures = futures
    
    
    print("Start training\n")
    time_start=time.time()
    rh.process_results(futures, raise_errors=False)
    print(f"Time elapsed: {time.time() - time_start}")
    date = datetime.datetime.now()
    print(f"{date.day}.{date.month}.{date.year}  {int(date.hour)+date_time_hour_offset}:{date.minute}:{date.second} \n")
    
    
    time.sleep(1)
    del futures, rh
    

    
## Dask Client
####################################################################################################################################### 
_Dask_Client = None


def create_dask_client(ip_port:str="127.0.0.1:8786"):
    """Erstelle einen Client mit einem gegebenen Scheduler.
    ip_port: Gebe IP-Adresse des Scheduler an. Standardwert: 127.0.0.1:8786 (lokal).
    """
    
    print(f"Client-IP: {ip_port}")
    _Dask_Client = _Client(ip_port)
    
    # Scatter Files 
    scatter_files(_Dask_Client)
    
    return _Dask_Client


def create_local_cuda_cluster():
    """Erstelle einen lokalen Dask-Cuda Cluster mit dask_cuda.
    """

    cluster = LocalCUDACluster()
    client = _Client(cluster)
    return client

# Auch für den lokalen Cluster anwendbar    
def scatter_files(client:_Client):
    """Verteilte pytorch_tools und pytorch_dispatcher_resulthandler im Cluster
    
    """
    client.upload_file('pytorch_dispatcher_resulthandler.py')
    client.upload_file('pytorch_tools.py')
    

    
# Pytorch DPP Model speichern und laden  
#######################################################################################################################################

########## Speichert auf Client
def ddp_save_model_dict(channel, model_dict, path, **kwargs):
    """Diese Methode speichert das Model als Dict.
    - .pt, wenn nur das Model gegeben ist.
    - .cpkt, wenn das Model mit weiteren Angaben wie Epoche und Optimierer übergeben wird. 
    Parameter:
    - channel: Publisher
    - model_dict: Model als Dict,
    - path: wo das Model gespeichert werden soll.
    - **kwargs: Sonstige Parameter wie z.B. optimizer_state_dict=optimizer.state_dict()
    """
    
    channel.put({"model_dict":              
                              {
                               'model_state_dict': model_dict,  # model.state_dict()
                               'path':             path,
                               'kwargs':           kwargs,
                              } }
                           )
    
    
    
def ddp_load_model():
    
    
    pass
    
    
# Sonstiges... 
#######################################################################################################################################


# Dask-Pytorch Dispatcher Methode von Saturncloud
# Open Source Project: https://github.com/saturncloud/dask-pytorch-ddp
def _get_worker_info(client: _Client) -> List[Dict]:
    workers = client.scheduler_info()["workers"]
    worker_keys = sorted(workers.keys())
    workers_by_host: Dict[str, List[str]] = {}
    for key in worker_keys:
        worker = workers[key]
        host = worker["host"]
        workers_by_host.setdefault(host, []).append(key)
    host = workers[worker_keys[0]]["host"]
    all_workers = []
    global_rank = 0
    for host in sorted(workers_by_host.keys()):
        local_rank = 0
        for worker in workers_by_host[host]:
            all_workers.append(
                dict(
                    worker=worker,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    host=host,
                )
            )
            local_rank += 1
            global_rank += 1
    return all_workers


def listDaskWorker(client:_Client):
    """Nutzt die Methode vom Dispatcher von Saturncloud um alle Worker aufzulisten.
    """
    return _get_worker_info(client)
   