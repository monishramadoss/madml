import requests
import platform
import multiprocessing
import psutil
import time

HOST_NAME = 'http://127.0.0.1:5000'
worker_data_frame = {
    "assembly": platform.machine(),
    "os": platform.processor(),
    "cpu_info": platform.processor(),
    "cpu_cores": multiprocessing.cpu_count(),
    "memory": psutil.virtual_memory().total
}

client_data_frame = {
    "user_id": 321,
    "progress": "new",
    "worker_id": -1
}


header = {'Content-Type': 'application/json', 'Accept': 'application/json'}
res = requests.get(HOST_NAME + '/worker_request/-1', headers=header, json=worker_data_frame)
WORKER_ID = res.text

while True:
    res = requests.get(HOST_NAME + "/worker_request/pull_layers/" + WORKER_ID, headers=header)
    model_json = res.json()
    client_id = model_json['client_id']
    if client_id != '-1':
        res = requests.get(HOST_NAME + "/worker_request/pull_weight/" + client_id, headers=header)
        weights = res.json()
        for layer in model_json['layers']:
            print(layer)
        for weight in weights['tensor_names']:
            res = requests.get(HOST_NAME + "/worker_request/get_weight/" + weight['weight'], headers=header)
            w = res.json()['weight']
            weight['weight'] = [float(x) for x in w] if w != [''] else 0
            print(weight)
        break
    time.sleep(10)
    break
requests.delete(HOST_NAME + '/worker_request/' + WORKER_ID, headers=header, json=worker_data_frame)
