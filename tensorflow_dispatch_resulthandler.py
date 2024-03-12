print("TF Dispatcher-Resulthandler 1.0")


import os
from typing import List, Callable, Any, Dict
from dask.distributed import Client
import json

from distributed import Pub, Sub
from typing import List, Dict
from distributed.utils import TimeoutError as DistributedTimeoutError
from distributed.client import wait, FIRST_COMPLETED, Future
import logging


# Dask-Pytorch Dispatcher
# Open Source Project: https://github.com/saturncloud/dask-pytorch-ddp
###########################################################################

def _get_worker_info(client: Client) -> List[Dict]:
    """
    returns a list of workers (sorted), and the DNS name for the master host
    The master is the 0th worker's host
    
    
    Das Ergebnis dieser Methode sieht so aus:
    
    'worker': 'IP:PORT',
    'local_rank': 0,      
    'global_rank': 0,
    'host': 'IP' ...
       
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
    tf_function: Callable,
    *args,
    backend: str = "nccl", 
    pass_local_rank: bool = False,
    **kwargs
):
    """
    Dispatch a pytorch function over a dask cluster, and returns a list of futures
    for the resulting tasks
    """

    all_workers = _get_worker_info(client)
    world_size = len(all_workers)
    
    port = 23456  # pick a free port?
    
    host = all_workers[0]["host"]
    futures = []
    
    for worker in all_workers:
        if pass_local_rank:
            fut = client.submit(
                dispatch_with_ddp,
                tf_function=tf_function,
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
            fut = client.submit(
                dispatch_with_ddp,
                tf_function=tf_function,
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                *args,
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        futures.append(fut)
    return futures



def set_worker_Ports(client:Client, ports:[int]=None):

    _Worker_keys = [] 
    _Worker_Ports = []
        
    worker_dict = _get_worker_info(client)
    

    if not ports:
            print(f"Keine Ports angegeben, generiere Ports für {len(worker_dict)} Worker. Ports können als Liste übergeben werden [3335, 6678, ...]")
            for i in range(len(worker_dict)):
                a = np.random.randint(20000, 41000)
                if a not in _Worker_Ports:
                    _Worker_Ports.append(a)
    else:
        _Worker_Ports = ports

    for i in range(len(worker_dict)):
        _Worker_keys.append(str(f"{worker_dict[i]['host']}:{_Worker_Ports[i]}"))

    with open(r'worker_ip_port.py', 'w') as fp:
        fp.write(f"Worker_keys={_Worker_keys}")
    client.upload_file("worker_ip_port.py")
   
    print(f"Worker und Worker Ports:\n{_Worker_keys}")


    
    
## Läuft auf jedem Worker
def dispatch_with_ddp(
    tf_function: Callable, 
    master_addr: Any,
    master_port: Any,
    rank: Any,
    world_size: Any,
    *args,
    backend: str = "nccl",
    **kwargs
) -> Any:
    
    import worker_ip_port
    # These are the parameters used to initialize the process group
    master_addr = str(master_addr)
    master_port = str(master_port)
    rank = str(rank)
    world_size = str(world_size)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size

    tf_mode = bool(os.environ.get('TF_PARAMETER_SERVER_MODE'))
    tf_mode = False

    tf_worker_types = ['worker', 'ps']
    tf_ps_server_rank = 0
    
    if tf_mode:
        print("Mode: PS")
        
        worker_list    = worker_ip_port.Worker_keys
        ps_list        = worker_ip_port.ps_keys
        ps_key_mapping = worker_ip_port.ps_mapping_keys
        ps_count       = worker_ip_port.ps_counts # mind 1.

        task_str = ""

        if int(rank) == 0:
             task_str = {"type": ps_key_mapping[int(rank)], "index": rank}        # Chief
        elif (int(rank) < ps_count) and (int(rank) != 0):
            task_str = {"type": ps_key_mapping[int(rank)], "index": int(rank)-1}         # PS
        else:
            task_str = {"type": ps_key_mapping[int(rank)], "index": int(int(rank) - ps_count)} 
        
        os.environ["TF_CONFIG"] = json.dumps({
                          "cluster": {
                                      "chief" : ps_list[0],
                                      "ps"    : ps_list[1:],
                                      "worker": worker_list, # ['149.201.182.188:29123', '149.201.182.203:23635', '149.201.182.205:38044'],
                                     },
                           "task": task_str 
                            })

        print(f"\nRank: {rank}\nTF_CONFIG:\n{os.environ.get('TF_CONFIG')}\n")

        try:
            u = 6+6
        finally:
            f=6
        return 0
        
    else:

        worker_list = worker_ip_port.Worker_keys
        os.environ["TF_CONFIG"] = json.dumps({
                          "cluster": {
                               "worker": worker_list, # ['149.201.182.188:29123', '149.201.182.203:23635', '149.201.182.205:38044'],
                                },
                            "task": {"type": "worker", "index": rank} 
                                 })

        print(f"\nRank: {rank}\nTF_CONFIG:\n{os.environ.get('TF_CONFIG')}\n")
    
        try:
            val = tf_function(*args, **kwargs)
        finally:
           # dist.destroy_process_group()
           a = 10
        return val




# Dask result handler
# Open Source Project: https://github.com/saturncloud/dask-pytorch-ddp
###########################################################################

class DaskResultHandler:
    """
    This class use Dask pubsub infra to pass intermediate results back from PyTorch
    jobs to the client.
    """

    def __init__(self, pub_sub_key:str="my_channel"):
        """Init Class
        Hier kann man zu Beginn Ordner oder Pfade festlegen.
        Ein Pfad, den man übergeben will, sollte so aussehen: "training/ordner1/ordner2/"
        
        Wenn nötig, können auch Funktionen ausgeführt werden, um Pfade, etc. zu erstellen. 
        
        trainingpath: Wo das Model gespeichert wird. 
        pub_sub_key:  Publisher/Subscriber Channel Name.
        _setup_working_dir: Erstellt den Ordner "training", wenn nicht vorhanden
        """
        
        self.pub_sub_key   = pub_sub_key

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
            
            
            if 'msg' in result:
                print(result['msg'])
            if 'model' in result:
                dict_data = result['model']

                model  = dict_data['model']
                format = dict_data['format']
                path   = dict_data['path']
                
                if (format == "weights"):  # Nur Gewichte
                    model.save_weights(f"{path}")
                if (format == "full"): # Ganzes Model, Endung .h5 oder .keras
                    model.save(f"{path}.h5")



     




















