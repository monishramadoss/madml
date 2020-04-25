import requests

HOST_NAME = 'http://127.0.0.1:5000'
client_data_frame = {
     "user_id": 321,
     "model_structure": [1, 2],
     "weights": [1, 2],
     "progress": "new",
     "worker_id": -1
}

requests.post(HOST_NAME + '/client_request', data=client_data_frame)
