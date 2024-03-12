print("TF tools Version 1.0")
TF_CPP_MIN_LOG_LEVEL = 1
CUDA_VISIBLE_DEVICES = 0
NCCL_SOCKET_NTHREADS = 4
NCCL_NSOCKS_PERTHREAD = 2
TF_PARAMETER_SERVER_MODE = "False"

import os
print(f"Setze Umgebungsvariablen:\n\
      TF_CPP_MIN_LOG_LEVEL: {TF_CPP_MIN_LOG_LEVEL}\n\
      CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}\n\
      NCCL_SOCKET_NTHREADS: {NCCL_SOCKET_NTHREADS}\n\
      NCCL_NSOCKS_PERTHREAD: {NCCL_NSOCKS_PERTHREAD}\n\
      TF_PARAMETER_SERVER_MODE: {TF_PARAMETER_SERVER_MODE}\n")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['NCCL_SOCKET_NTHREADS']  = str(NCCL_SOCKET_NTHREADS) 
os.environ['NCCL_NSOCKS_PERTHREAD'] = str(NCCL_NSOCKS_PERTHREAD)
os.environ['TF_PARAMETER_SERVER_MODE'] = str(TF_PARAMETER_SERVER_MODE)



from distributed import Client as _Client
import tensorflow as tf
from typing import List, Dict
import numpy as np
import time
import datetime

import tensorflow_dispatch_resulthandler as _module_tdr

## Dask
############################################################################################################
def get_dask_client(ip_port:str='127.0.0.1:8786'):
    """Erstelle Dask Client
    ip_port: Adresse des Dask Schedulers
    """
    _dask_client = _Client(ip_port)
    return _dask_client


def _get_worker_info(client: _Client) -> List[Dict]:
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
    
############################################################################################################

## Schreibe datei mit Worker IP und Ports, Scatter Datei
def _set_worker_Ports(client:_Client, ports:[int]=None, ps:int=2): # Erste n-Worker sind PS. Mind. 1

    _Worker_keys  = []   # Nach Rang
    _Worker_Ports = []
    
    _worker_ps = []
    _worker_ps_mapping = {}

        
    worker_dict = _get_worker_info(client)

    tf_mode = bool(os.environ.get('TF_PARAMETER_SERVER_MODE'))
    tf_mode = False

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

    # Mapping PS
    if tf_mode:
        for i in range(len(_Worker_keys)):
            if i == 0:  # chief
                 _worker_ps_mapping[i] = 'chief'  
                 _worker_ps.append(_Worker_keys[i])
            elif (i < ps) and (i != 0):
                _worker_ps.append(_Worker_keys[i])
                _worker_ps_mapping[i] = 'ps'       # Rang = Job
            else:
                _worker_ps_mapping[i] = 'worker'
                
        _Worker_keys = _Worker_keys[ps:]

         
    with open(r'worker_ip_port.py', 'w') as fp:
        fp.write(f"Worker_keys={_Worker_keys}\n")
        fp.close()
    if tf_mode:
         with open(r'worker_ip_port.py', 'a') as fp: 
             fp.write(f"ps_keys={_worker_ps}\n")
             fp.write(f"ps_mapping_keys={_worker_ps_mapping}\n")
             fp.write(f"ps_counts={ps}\n")
             fp.close()
    client.upload_file("worker_ip_port.py")
   
    print(f"Worker und Worker Ports für TF (Ports zwischen 20000-41000):\n{_Worker_keys}\n")
    if tf_mode:
        print(f"PS List: {_worker_ps}, Mapping to use: {_worker_ps_mapping}")


## Start Training
########################################################   
def run(train:callable, client, *args, date_time_hour_offset:int=0, **kwargs):
    """Starte das Training.
    train: Trainingsfunktion
    client: Dask Client
    date_time_hour_offset: Rechne +/- zur ausgegebenen Systemzeit
    **kwargs: Übergebe der Trainingsfunktion Parameter, wie: a=100,  data={...}
    """

    workers = client.has_what().keys()
    n_workers = len(workers)
    print(f"Worker count: {n_workers}")

    client.wait_for_workers(n_workers)

    rh = _module_tdr.DaskResultHandler("my_channel")   
    
    #########
    
    futures = _module_tdr.run(client, train, **kwargs)
    dask_futures = futures
    
    
    print("Start training\n")
    time_start=time.time()
    rh.process_results(futures, raise_errors=False)
    print(f"Time elapsed: {time.time() - time_start}")
    
    date = datetime.datetime.now()
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"{date.day}.{date.month}.{date.year}  {date.hour}:{date.minute}:{date.second}\n")
    
    time.sleep(1)
    del futures, rh



def init(client:_Client, ports:[int]=None, ps_count:int=2):
    """Setze Ports der Worker und lade Module im Cluster hoch. 
    client: Dask Client
    ports: Die Ports als Liste die genutzt werden sollen, sonst wird eine generiert mit Ports zwischen 20000-41000
    """
    print(f"Init: Worker IP und Port, Umgebungsvariablen, Scatter Module")

    client.upload_file("tensorflow_dispatch_resulthandler.py")     # Scatter .py Datei 
    client.upload_file("tensorflow_tools.py")                      
    _set_worker_Ports(client, ports=ports, ps=ps_count)                         # Setze Ports

     
    


## Tensorflow Strategie ## 
##############################################################################################################
# strategie_MultiWorkerMirrored # 
def strategie_MultiWorkerMirrored(option:str="NCCL") -> tf.distribute.MultiWorkerMirroredStrategy:
     """ Gebe die Strategie zurück.
     option: NCCL: GPU, RING: CPU,  AUTO: Automatisch

     """
     if option not in ["NCCL", "RING", "AUTO"]:
         raise ValueError("Wähle aus [NCCL, RING, AUTO]")
         
     communication_options = tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL) 
     #strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
     return tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)


## Save / Load Model
##############################################################################################################

def save_model(channel, model, path, format):
    """Speichere Model
    channel: Publisher Subscriber Channel
    model: Keras Model
    path: Speicherpfad
    format:str: ['weights', 'full']

    Lese mehr: https://www.tensorflow.org/guide/keras/serialization_and_saving
    """
    
    channel.put({ 'model': {
                         'model':  model,  
                         'format': format,
                         'path'  : path
                     }})  



## TF Checkpoint ## 
##############################################################################################################
def _is_chief(task_type, task_id):
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def _write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

##  Model Speichern Checkpoint ##
############################################################



## Speichere Model Multi-Worker 
def model_save__multiworker(model, strategy:tf.distribute.MultiWorkerMirroredStrategy, dir:str= '/tmp/keras-model'):
    """Speichere Model Checkpoint.
    model: Model
    strategy: Erstellte Strategie
    checkpoint_dir: Pfad, default: '/tmp/keras-model'

    Lese mehr in: "Model saving and loading": https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    """
    task_type, task_id = (strategy.cluster_resolver.task_type,strategy.cluster_resolver.task_id)
    write_dir = _write_filepath(dir, task_type, task_id)
    model.save(write_dir)
    print(f"- Model Multi-Worker save at {write_dir}")

## Checkpoint saving and restoring
def ckpt_model_saving_and_restoring__create(model, strategy:tf.distribute.MultiWorkerMirroredStrategy, checkpoint_dir:str= '/tmp/keras-model', to_keep:int=1):
    """
    'Here, you'll create one Checkpoint that tracks the model, which is
     managed by the tf.train.CheckpointManager, so that only the latest checkpoint is preserved'

    model: Model
    strategy: Erstellte Strategie
    checkpoint_dir: Pfad, default: '/tmp/keras-model'
    Lese mehr in: "Checkpoint saving and restoring": https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    """
    checkpoint = tf.train.Checkpoint(model=model)
    
    task_type, task_id = (strategy.cluster_resolver.task_type,strategy.cluster_resolver.task_id)
    write_checkpoint_dir = _write_filepath(checkpoint_dir, task_type, task_id)
    print(f"\n-  Ckpt: saving_and_restoring: {write_checkpoint_dir}  -\n")
    
    checkpoint_manager = tf.train.CheckpointManager(
         checkpoint, directory=write_checkpoint_dir, max_to_keep=to_keep)
    print(f"Create -saving_and_restoring- Checkpoint at {write_checkpoint_dir}")

    return checkpoint_manager, checkpoint

## Speichere Model 
def ckpt_model_saving_and_restoring__save(checkpoint_manager:tf.train.CheckpointManager,  strategy:tf.distribute.MultiWorkerMirroredStrategy, checkpoint_dir:str='/tmp/keras-model'):
    """Speichere Checkpoint und lösche auf andere Worker. 
    checkpoint_manager: Der Erstellte Checkpointmanager
    trategy: Erstellte Strategie
    checkpoint_dir: Pfad, default: '/tmp/keras-model'
    """
    checkpoint_manager.save()
    
    task_type, task_id = (strategy.cluster_resolver.task_type,strategy.cluster_resolver.task_id)  # Für _is_chief()
    write_checkpoint_dir = _write_filepath(checkpoint_dir, task_type, task_id)
    
    if not _is_chief(task_type, task_id):
       tf.io.gfile.rmtree(write_checkpoint_dir)
    print(f"Save -saving_and_restoring- at {write_checkpoint_dir}")

## Model wiederherstellen
def ckpt_model_saving_and_restoring__load(model, strategy:tf.distribute.MultiWorkerMirroredStrategy, checkpoint_dir:str='/tmp/keras-model'):
    """Lade den aktuelsten Checkpoint der mit ckpt_model_saving_and_restoring__save() erstellt wurde.
    model: Model
    dir: Pfad, default: '/tmp/keras-model'
    strategy: Erstellte Strategie
    checkpoint_dir: Pfad, default: '/tmp/keras-model'
    Lese mehr in: "Checkpoint saving and restoring": https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    """
    checkpoint = tf.train.Checkpoint(model=model)
    
    task_type, task_id = (strategy.cluster_resolver.task_type,strategy.cluster_resolver.task_id)
    dir = _get_temp_dir(checkpoint_dir, task_id)
    
    print(f"\n -  Load Checkpoint from: {dir}  -\n")
    latest_checkpoint = tf.train.latest_checkpoint(dir)
    checkpoint.restore(latest_checkpoint)

    return checkpoint

    print(f"Load -saving_and_restoring- from {dir}")


## Für Fehlertoleranz
def ckpt__BackupAndrestore(strategy:tf.distribute.MultiWorkerMirroredStrategy, backup_dir:str="/tmp/backup", freq="epoch"):
    """Speichere Zustand des Models. Bei Unterbrechnungen kann das Model wiederhergestellt werden.
    strategy: Erstellte Strategie
    backup_dir: Pfad, default: "backup_dir"
    freq: wenn int: Speichere alle n-Batches. Wenn String 'epoch', speichere jede Epoche
    Lese mehr in: "The BackupAndRestore callback": https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#model_saving_and_loading
    """
    #task_type, task_id = (strategy.cluster_resolver.task_type,strategy.cluster_resolver.task_id)
    #backup_dir = _write_filepath(dir, task_type, task_id)
    
    print(f"\nCkpt -BackupAndrestore- Backup at {backup_dir}\n")
    callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir, save_freq=freq)]
    
    return callbacks



# Sonstiges
##############################################################################################################
    
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
