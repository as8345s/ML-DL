"""
Autor: Alexander Schechtel
FH-Aachen FB5 Informatik.

"""

print("Rapids-tools 1.0")
print("Umgebungsvariablen werden gesetzt")
NCCL_SOCKET_NTHREADS   = '4'
NCCL_NSOCKS_PERTHREAD  = '2'
import os
os.environ['NCCL_SOCKET_NTHREADS']  = NCCL_SOCKET_NTHREADS
os.environ['NCCL_NSOCKS_PERTHREAD'] = NCCL_NSOCKS_PERTHREAD
print(f"NCCL_SOCKET_NTHREADS: {NCCL_SOCKET_NTHREADS}\nNCCL_NSOCKS_PERTHREAD: {NCCL_NSOCKS_PERTHREAD}")


## Imports
######################################################
## Dask
import dask 
from dask.distributed import Client, wait

## Rapids 
from cuml.dask.cluster.kmeans import KMeans as multi_kmeans
from cuml import KMeans as single_Kmeans

# Numpy
from numpy.testing import assert_equal              as _assert_equal
from numpy.testing import assert_array_almost_equal as _assert_array_almost_equal
import numpy

## Sklearn
from sklearn.cluster import KMeans as _sk_Kmeans

# Sonstiges
import cupy
import cudf


    
## Dask Client
######################################################## 
def create_dask_client(ip_port:str="127.0.0.1:8786"):
    
    print(f"Client IP: {ip_port}")
    _Dask_Client = Client(ip_port)
    
    # Scatter Files 
    scatter_files(_Dask_Client)
    
    return _Dask_Client

    
def scatter_files(client:Client):
    client.upload_file('rapids_tools.py')
    
    

## Numpy
#######################################################
# - Teste auf Gleichheit

def arrays_equal(X, y, printresult:bool=True, rows:int=10):
   
    if len(X) != len(y):
        print("Arrays nicht gleich lang")
        return
              
    # Ausgabe der Ungleichiet... dann return  
    if printresult:
        print(f"Gebe {rows} Zeilen aus, die ungleich sind")
        i = 0
        for i in range(len( X )):
            if X[i] != y[i]:
                print(f"X: { X[i]} \t!= \t y: {y[i]} ")
            if i == rows:
                break
            i+=1
    
        
    return _assert_equal(X, y)
    
    
    

def arrays_almost_equal (X, y, prec:int=4, printres:bool=True, rows:int=10):
       
    if len(X) != len(y):
        print(f"Arrays nicht gleich lang, X: {X} y: {y}")
        return
              
    # Ausgabe der Ungleichiet... dann return  
    if printresult:
        print(f"Gebe {rows} Zeilen aus, die ungleich sind.")
        i = 0
        for i in range(len( X )):
            if X[i] != y[i]:
                print(f"X: { X[i]} \t!= \t y: {y[i]} ")
            if i == rows:
                break
            i+=1
    
        
    return _assert_array_almost_equal(X, y, decimal=prec) 
    



## MNMG K-means tools
################################################################################################################

# - SNSG Model zu MNMG Model - #
def mnmg_kmeans__singleModel_to_multiModel(client:Client, loaded_single_gpu_model:single_Kmeans , distributed_model:multi_kmeans):
    print(f"K-Means: Überführe cuMl SNSG zu cuML MNMG.")

    # Übergebe Cluster
    distributed_model.client = client

    distributed_model.verbose = False
    #distributed_model.kwargs = loaded_single_gpu_model.kwargs

    # Notwendig, distributed.client.Future
    distributed_model.internal_model  = loaded_single_gpu_model  

    distributed_model.datatype        = 'cudf'
    distributed_model.output_type     = loaded_single_gpu_model.output_type
    distributed_model._input_type     = loaded_single_gpu_model._input_type
    distributed_model._input_mem_type = None
    distributed_model.target_dtype    = None
    distributed_model.n_features_in_  = loaded_single_gpu_model.n_features_in_
    distributed_model.n_clusters      = loaded_single_gpu_model.n_clusters
    distributed_model.random_state    = loaded_single_gpu_model.random_state
    distributed_model.max_iter        = loaded_single_gpu_model.max_iter
    distributed_model.tol             = loaded_single_gpu_model.tol
    distributed_model.n_init          = loaded_single_gpu_model.n_init
    distributed_model.inertia_        = loaded_single_gpu_model.inertia_
    distributed_model.n_iter_         = loaded_single_gpu_model.n_iter_
    distributed_model.oversampling_factor   = loaded_single_gpu_model.oversampling_factor
    distributed_model.max_samples_per_batch = loaded_single_gpu_model.max_samples_per_batch
    distributed_model.init          = loaded_single_gpu_model.init
    distributed_model._params_init  = loaded_single_gpu_model._params_init
    distributed_model.n_rows        = loaded_single_gpu_model.n_rows
    distributed_model.n_cols        = loaded_single_gpu_model.n_cols
    #distributed_model.handle
    distributed_model.output_mem_type = loaded_single_gpu_model.output_mem_type
    distributed_model.labels_         = loaded_single_gpu_model.labels_
    distributed_model.cluster_centers_ = loaded_single_gpu_model.cluster_centers_
    distributed_model.dtype            = loaded_single_gpu_model.dtype
    
    return distributed_model
    
    

# - cuML Model -> Sklearn Model - #
def mnmg_kmeans__singleModel_to_sklearnModel(single_gpu_model:single_Kmeans, sk_model_new:_sk_Kmeans):
    print(f"K-Means: Überführe cuML Model zu Sklearn Model.")
    
    #sk_model_new =  sk_Kmeans()

    sk_model_new.n_clusters =  single_gpu_model.n_clusters
    sk_model_new.init       =  single_gpu_model.init
    sk_model_new.max_iter   =  single_gpu_model.max_iter
    sk_model_new.tol        =  single_gpu_model.tol
    sk_model_new.random_state = single_gpu_model.random_state

    #sk_model_new.copy_x     = single_gpu_model.copy_x              Kein copy_x
    sk_model_new.copy_x=True

    #sk_model_new.algorithm  = single_gpu_model.algorithm           Kein algorithm
    sk_model_new.n_features_in_ = single_gpu_model.n_features_in_
    #sk_model_new._tol       = single_gpu_model._tol                Kein _tol
    #sk_model_new._n_init    = single_gpu_model._n_init             Kein _n_init
    #sk_model_new._algorithm = single_gpu_model._algorithm          kein _algorithm

    #sk_model_new._n_threads = single_gpu_model._n_threads          kein _n_threads
    sk_model_new._n_threads  = 1

    sk_model_new.cluster_centers_ = cupy.asnumpy(cudf.DataFrame.to_cupy(single_gpu_model.cluster_centers_)) 
    #sk_model_new._n_features_out = single_gpu_model._n_features_out   Kein _n_features_out
    sk_model_new.labels_    = cupy.asnumpy(cudf.DataFrame.to_cupy(single_gpu_model.labels_))
    sk_model_new.inertia_   = single_gpu_model.inertia_
    sk_model_new.n_iter_    = single_gpu_model.n_iter_

    return sk_model_new


# - cuML Model -> Sklearn Model - #
def snsg_kmeans__singleModel_to_sklearnModel(single_model:single_Kmeans, sk_model_new:_sk_Kmeans):
    print(f"Überführe cuML Model zu Sklearn Model.")
    
    #sk_model_new =  sk_Kmeans()

    sk_model_new.n_clusters =  single_model.n_clusters
    sk_model_new.init       =  single_model.init
    sk_model_new.max_iter   =  single_model.max_iter
    sk_model_new.tol        =  single_model.tol
    sk_model_new.random_state = single_model.random_state

    #sk_model_new.copy_x     = single_model.copy_x              Kein copy_x
    sk_model_new.copy_x=True

#sk_model_new.algorithm  = single_model.algorithm           Kein algorithm
    sk_model_new.n_features_in_ = single_model.n_features_in_
    #sk_model_new._tol       = single_model._tol                Kein _tol
    #sk_model_new._n_init    = single_model._n_init             Kein _n_init
    #sk_model_new._algorithm = single_model._algorithm          kein _algorithm

    #sk_model_new._n_threads = single_model._n_threads          kein _n_threads
    sk_model_new._n_threads  = 1

    sk_model_new.cluster_centers_ = single_model.cluster_centers_.get()
    #sk_model_new._n_features_out = single_model._n_features_out   Kein _n_features_out
    sk_model_new.labels_    = single_model.labels_.get()
    sk_model_new.inertia_   = single_model.inertia_
    sk_model_new.n_iter_    = single_model.n_iter_
    
    return sk_model_new
    


## SNSG KNN tools
################################################################################################################


# - cuML Model -> Sklearn Model - #
def snsg_knn_c__singleModel_to_sklearnModel(single_model:single_Kmeans, sk_model_new:_sk_Kmeans):
    """Params:
    single_model: SNSG cuML Model
    sk_model_new: Basis Sklearn model
    """
    print(f"Überführe cuML Model zu Sklearn Model.")
    
    sk_model_new.n_neighbors = single_model.n_neighbors
    #sk_model_new.radius      = single_model. Kein radius
    sk_model_new.algorithm   = single_model.algorithm
    #sk_model_new.leaf_size   = single_model.leaf_size   Keine leaf_size
    sk_model_new.metric      = single_model.metric
    sk_model_new.metric_params = single_model.metric_params
    sk_model_new.p           = single_model.p
    #sk_model_new.n_jobs      = single_model.n_jobs      Kein n_jobs
    sk_model_new.weights     = single_model.weights
    sk_model_new.n_features_in_ = single_model.n_features_in_
    #sk_model_new.outputs_2d_ =    single_model.outputs_2d_   Kein outputs_2d_

    sk_model_new.outputs_2d_ = False

    sk_model_new.classes_    =    single_model.classes_.get()  # Hier muss auch ein .get()
    sk_model_new.n_neighbors =    single_model.n_neighbors

    sk_model_new._y          =    single_model.y.get()

    sk_model_new.effective_metric_params_ = single_model.effective_metric_params_
    sk_model_new.effective_metric_        = single_model.effective_metric_

    sk_model_new._fit_method =  single_model._fit_method
    sk_model_new._fit_X      =  single_model._fit_X.get()


    sk_model_new.n_samples_fit_ = single_model.n_samples_fit_
    #sk_model_new._tree          = single_model.tree     Kein tree
    
    return sk_model_new
    


# - cuML Model -> Sklearn Model - #
def snsg_knn_r__singleModel_to_sklearnModel(single_model:single_Kmeans, sk_model_new:_sk_Kmeans):
    """Params:
    single_model: SNSG cuML Model
    sk_model_new: Basis Sklearn model
    """
    print(f"Überführe cuML Model zu Sklearn Model.")
    
    sk_model_new.n_neighbors  = single_model.n_neighbors
    #sk_model_new.radius      = single_model. kein radius
    sk_model_new.algorithm    = single_model.algorithm
    #sk_model_new.leaf_size   = single_model. kein leaf_size
    sk_model_new.metric       = single_model.metric
    sk_model_new.metric_params = single_model.metric_params
    sk_model_new.p             = single_model.p
    #sk_model_new.n_jobs       = single_model. kein n_jobs
    sk_model_new.weights       = single_model.weights
    sk_model_new.n_features_in_ = single_model.n_features_in_
    sk_model_new._y             = single_model.y.get()
    sk_model_new.effective_metric_params_ = single_model.effective_metric_params_
    sk_model_new.effective_metric_        = single_model.effective_metric_

    sk_model_new._fit_method = single_model._fit_method # siehe API was passt

    sk_model_new._fit_X      = single_model._fit_X.get()
    sk_model_new.n_samples_fit_ = single_model.n_samples_fit_
    #sk_model_new._tree        = single_model._ kein _tree
    
    return sk_model_new