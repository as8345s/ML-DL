"""
Autor: Alexander Schechtel
FH-Aachen FB5 Informatik.

"""
print("PyTorch Dispatcher-/Resulthandler 1.0")


from typing import List, Callable, Any, Dict
from dask.distributed import Client
import torch.distributed as dist
import socket
import torch 
from distributed.pubsub import Pub, Sub
from typing import List, Optional, Dict
from distributed.client import wait, FIRST_COMPLETED, Future
from os.path import join, exists, dirname
from distributed.utils import TimeoutError as DistributedTimeoutError
import logging
import os

from torch.distributed.rpc import init_rpc, rpc_async, rpc_sync, remote
import torch.distributed.rpc as rpc


# Dask-Pytorch Dispatcher
# Open Source Project: https://github.com/saturncloud/dask-pytorch-ddp
###################################################################################################


from typing import List, Callable, Any, Dict
from dask.distributed import Client
import torch.distributed as dist
from distributed.pubsub import Pub, Sub
import socket
import time

def _get_worker_info(client: Client) -> List[Dict]:
    """
    returns a list of workers (sorted), and the DNS name for the master host
    The master is the 0th worker's host
    
    
    Das Ergebnis dieser Methode sieht so aus:
    
    'worker': 'IP:PORT',
    'local_rank': 0,      
    'global_rank': 0,
    'host': 'IP' ...
    
    Beispiel: 2 Node mit je 2 GPUs
       Node 1:
           GPU 1:    Lokal 0   
           GPU 2:    Lokal 1   
       Node 2:
           GPU 1:    Lokal 0   
           GPU 2:    Lokal 1  
           
       
    """
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


def run(
    client: Client,
    pytorch_function: Callable,
    pytorch_mode:str,  # New
    *args,
    backend: str = "nccl",  # nccl: gpu | gloo: cpu
    pass_local_rank: bool = False,
    **kwargs
):
    """
    Dispatch a pytorch function over a dask cluster, and returns a list of futures
    for the resulting tasks
    """
    all_workers = _get_worker_info(client)
    world_size = len(all_workers)
    
    
    port = os.getenv('PYTORCH_DIST_DDP_PORT')  # pick a free port?
    
    host = all_workers[0]["host"]
    futures = []
    for worker in all_workers:
        if pass_local_rank:
            fut = client.submit(
                dispatch_with_ddp,
                pytorch_function=pytorch_function,
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                *args,
                local_rank=worker["local_rank"],
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        else:
            this_host= str(worker["host"])
            fut = client.submit(
                dispatch_with_ddp,
                pytorch_function=pytorch_function,
                pytorch_mode=pytorch_mode,  # Added
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                worker_host=this_host,   # added 
                *args,
                local_rank=worker["local_rank"],  # added
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        futures.append(fut)
    return futures


# pylint: disable=too-many-arguments
def dispatch_with_ddp(
    pytorch_function: Callable,
    pytorch_mode:str,
    master_addr: Any,
    master_port: Any,
    rank: Any,
    world_size: Any,
    worker_host: Any,   # added
    *args,
    local_rank:int=0,  # added
    backend: str = "nccl",  
    **kwargs
) -> Any:
    
    """ Konfiguration von PyTorch DDP.
    
    runs a pytorch function, setting up torch.distributed before execution
    and tearing it down afterwards.
    
    Konfiguration von PyTorch DDP. 
   
    Diese Funktion wird von ALLEN worker ausgeführt. 
    
    Damit alles gut geht müssen alle Einstellungen passen. Im Normalfall muss hier nichts verändert werden.
    
    pytorch_function:    Deine Trainingsfunktion.
    
    # Meist worker 0, kann auch Worker 3 sein. Siehe run()
    master_addr:         Master IP für DDP
    master_port:         Master Port für DDP
    rank:                Globaler Rang des Workers 
    
    backend:             Backend für PyTorch DDP.:  nccl: GPU,  gloo: CPU
    
    """
    
   # if pytorch_mode =="model_parallel":
        #    res = _use_RPC(master_addr, master_port, rank, world_size)
         #   return res
        
        
    
    if "PYTORCH_MODULE_LOG" in os.environ:
        PYTORCH_MODULE_LOG = bool(os.getenv('PYTORCH_MODULE_LOG'))

    if PYTORCH_MODULE_LOG:
        print(f"\nInfo| IP:{worker_host} Global Rank:{rank}, Local Rank:{local_rank}, Mode: {pytorch_mode}")
    
    # These are the parameters used to initialize the process group
    master_addr = str(master_addr)
    master_port = str(master_port)
    rank = str(rank)
    world_size = str(world_size)
    

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size
    os.environ["WORKER_HOST"] = worker_host
    
    backend = os.getenv('PYTORCH_DIST_BACKEND')
    
   
    ### PyTorch RPC-RRef Model-Parallel ### ### ### ### ### ###
    if pytorch_mode =="model_parallel":
            # In Tools
            #os.environ["GLOO_SOCKET_IFNAME"] = "eno1"  # unable to find Adress for eno4   etho zb, Netzwerkadresse
            #os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # - # Lösung: https://github.com/pytorch/tensorpipe/issues/201 
            #os.environ["TP_SOCKET_IFNAME"]   = "eno1"
            
           # val = _use_RPC(master_addr, master_port, rank, world_size )
            #return val
            
           # """
            start_on_rank  = int(os.getenv('PYTORCH_MODEL_PARALLEL__START_ON_RANK'))
            
            gpu_map = {}
            # Erstelle GPU map
            for i in range(int(world_size)):
                gpu_map[f'w{i}'] = {0: 0}
            del gpu_map[f'w{rank}']    # Lösche eigenen Eintrag 
            print(f"RPC: gpu map: {gpu_map}")
            
            ## RPC Backend Optionen
           
            options=rpc.TensorPipeRpcBackendOptions(
               num_worker_threads=18,
               rpc_timeout=60,
               init_method=f'tcp://{master_addr}:{master_port}', device_maps=gpu_map  
            )

            print(f"check rank==start_on_rank: {int(rank)==int(start_on_rank)}, rank:{type(rank)}, start_on_rank:{type(start_on_rank)}")
            val=100
            try: 
                
                ## Wo die Trainingsfunktion gestartet werden soll ##
                if int(rank)==1:  # Siehe Tools
                    rpc.init_rpc(f"w{rank_int}", rank=int(rank), world_size=int(world_size), rpc_backend_options=options, backend=rpc.BackendType.TENSORPIPE)            
                    print(f"Init master, rank: {int(rank)}, RPC name: w{rank_int}") 
                    val = pytorch_function(*args, **kwargs)
                else:
                    print(f"init worker, rank: {rank_int}, RPC name: w{rank_int}")
                    rpc.init_rpc(f"w{int(rank)}", rank=int(rank), world_size=int(world_size), rpc_backend_options=options, backend=rpc.BackendType.TENSORPIPE) 
                        
            finally:
                print("Bevore Shutdown RPC")
                time.sleep(3)
                rpc.shutdown()
                print("Shutdown RPC")
            return val
            #"""
     ### ### PyTorch RPC-RRef Model-Parallel Ende ### ### ### ### ### ###
    

    try:
        # Erstelle eine Prozessgruppe für DDP
        # Der Name ist optional, mit dem Namen kann man auf die Gruppe zugreifen
        
        dist.init_process_group(backend=backend, group_name="my_group")  #bucket_cap_mb=25 (mb)
        val = pytorch_function(*args, **kwargs)
    finally:
        dist.destroy_process_group()
       
    return val



##################################
def _use_RPC(master_addr, master_port, rank, world_size):
     pass
    


# Dask result handler
# Open Source Project: https://github.com/saturncloud/dask-pytorch-ddp
####################################################################################################################

class DaskResultHandler:
    """
    This class use Dask pubsub infra to pass intermediate results back from PyTorch
    jobs to the client.
    """

    def __init__(self, pub_sub_key:str="my_channel", trainingpath:str="template_pytorch/"):
        """Init Class
        Hier kann man zu Beginn Ordner oder Pfade festlegen.
        Ein Pfad, den man übergeben will, sollte so aussehen: "training/ordner1/ordner2/"
        
        Wenn nötig, können auch Funktionen ausgeführt werden, um Pfade, etc. zu erstellen. 
        
        trainingpath: Wo das Model gespeichert wird. 
        pub_sub_key:  Publisher/Subscriber Channel Name.
        _setup_working_dir: Erstellt den Ordner "training", wenn nicht vorhanden
        """
        
        self.training_path = trainingpath
        self.pub_sub_key   = pub_sub_key
        #self._setup_working_dir(trainingpath)
        
        # Erstelle Pfade, sonstige Vorbereitungen... 
        # _setup_working_dir()

        
    @classmethod
    def _get_all(cls, sub: Sub):
        """Auslesen des Channels:
        Geben die Daten zurück die jede Epoche von einem Worker hochgeladen werden. 
        - Host ist meist Subscriber und hört allen Topics zu.
        - Es kann mehrere Nachrichten und Channels geben.
        """
        while True:
            try:
                yield sub.get(timeout=1.0)
            except DistributedTimeoutError:
                break
                

    def _get_results(self, futures: List[Future], raise_errors: bool = True):
        """Get Dask results.
        Hier erstellen wir ein Subscriber sub_stats, der die Daten aus dem Channel pub_sub_key auslesen soll. 
        
        """    
        sub_stats = Sub(self.pub_sub_key)  

        
        while True: 
            
            # Für Subscriber, get Channel data.
            for obj in self._get_all(sub_stats):     
                yield obj
            
            # keine Futures? => break. 
            if not futures:
                break
                
            try:
                # Dask:   wait(fs[, timeout, return_when])      Wait until all/any futures are finished
                # Read here: https://distributed.dask.org/en/stable/api.html
                result = wait(futures, 5, FIRST_COMPLETED)  #0.1
            except DistributedTimeoutError:
                continue

            for fut in result.done:     
                try:                   
                    fut.result()  
                    
                except Exception as e:  # pylint: disable=broad-except
                    logging.exception(e)
                    
                    if raise_errors:
                        raise
                        
            futures = result.not_done


    def process_results(self, futures: List[Future], raise_errors: bool = True) -> None:
        
        """Verarbeitung der Daten der Futures die von Dask geliefert werden.
        Die Ergebnisse kommen als Liste an.
        
        Die Liste "futures" enthält alles, was wir mit dem Publisher hochladen.
        Das kann das Model sein (als Dict) und weitere Werte wie Loss, Acc, ...
        
        Hier kann eine Bedingung eingefügt werden, um das Training zu stoppen und das Model zu speichern. 
        
        Mit torch.save wird das Model gespeichert. Oder implementiere eine eigene Methode für das Speichern. 
        """

        for result in self._get_results(futures, raise_errors=raise_errors):
            
            
            
            if "msg" in result:
                msg = result['msg']
                print(msg)
            
            if "model_dict" in result:
                model_dict_data = result['model_dict']
                
                path   = model_dict_data['path']
                kwargs = model_dict_data['kwargs']
                
                
                if len(kwargs.keys()) !=0 :   # Checkpoint
                    torch.save({
                        'model_state_dict': model_dict_data['model_state_dict'],
                        'kwargs':           kwargs
                         }, str(path+".ckpt"))
                        
                else:                         # .pt oder .pth 
                    torch.save(model_dict_data['model_state_dict'], str(path+".pt")) 
                               
 
    def _setup_working_dir(self, trainingpath:str):
        """Create Dir.
        Wenn es kein Ordner mit dem Namen gibt, wird einer erstellt.  
        """
        if not exists(dirname(trainingpath)):
                os.makedirs(dirname(trainingpath))
                