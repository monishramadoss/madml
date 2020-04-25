import requests
import platform
import multiprocessing
import psutil

HOST_NAME = 'http://127.0.0.1:5000'
worker_id = 456
worker_data_frame = {
    "worker_id": worker_id,
    "assembly": platform.machine(),
    "os": platform.processor(),
    "cpu_info": platform.processor(),
    "cpu_cores": multiprocessing.cpu_count(),
    "memory": psutil.virtual_memory().total
}

client_data_frame = {
     "user_id": -1,
     "model_structure": [],
     "weights": [],
     "progress": "",
     "worker_id": -1
}

requests.post(HOST_NAME + '/ready_worker', data=worker_data_frame)
res = requests.get(HOST_NAME + '/ready_worker').json()
client_data_frame = dict([(k, res[k]) for k in client_data_frame.keys()])
client_data_frame['worker_id'] = worker_id
client_data_frame['progress'] = 'working'
res = requests.put(HOST_NAME + "/ready_worker", data=client_data_frame)
