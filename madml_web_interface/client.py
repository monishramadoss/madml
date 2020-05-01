import requests
import json

import onnx
from google.protobuf.json_format import MessageToJson

HOST_NAME = 'http://127.0.0.1:5000'
SUCCESS_FULL_CONNECTION = False
TEST_ONNX = True
client_data_frame = {
    "user_id": 321,
    "progress": "new",
    "worker_id": -1
}

weight_payload = {
    'user_id': -1,
    'weight': [],
    'parameters': [],
    'layer': '',
}

layer_payload = {
    'user_id': -1,
    'op': '',
    'attributes': '',
    'input': [],
    'output': [],
    'name': '',
}

res = requests.get(HOST_NAME + '/client_request', data=client_data_frame)
SUCCESS_FULL_CONNECTION = res.status_code == 200
weight_payload['user_id'] = client_data_frame['user_id']
layer_payload['user_id'] = client_data_frame['user_id']

if TEST_ONNX:
    model_path = "./tests/mobilenetv2-1.0.onnx"
    onnx_model = onnx.load(model_path)
    onnx_json = json.loads(MessageToJson(onnx_model))
    graph = onnx_json['graph']

    weight_payloads = list()
    for x in graph['initializer']:
        if x['dataType'] == 1:
            weight_payload['weight'] = x['floatData']
        weight_payload['layer'] = x['name']
        weight_payload['parameters'] = x['dims']
        weight_payloads.append(weight_payload)

    layer_payloads = list()
    for x in graph['node']:
        layer_payload['input'] = x['input']
        layer_payload['output'] = x['output']
        layer_payload['name'] = x['name']
        if 'attribute' in x.keys():
            layer_payload['attributes'] = x['attribute']
        layer_payload['op'] = x['opType']
        layer_payloads.append(layer_payload)

    for x in graph['input']:
        pass
    for x in graph['output']:
        pass

    res = requests.post(HOST_NAME + '/client_request', data=graph)

