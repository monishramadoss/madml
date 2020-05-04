import requests
import json
import tempfile

import onnx
from google.protobuf.json_format import MessageToJson

HOST_NAME = 'http://127.0.0.1:5000'
SUCCESS_FULL_CONNECTION = False
TEST_ONNX = True
CLIENT_ID = 321

client_data_frame = {
    "progress": "new",
    "worker_id": -1
}

weight_payload = {
    'weight': [],
    'dims': [],
    'layer': '',
}

layer_payload = {
    'opType': '',
    'attributes': '',
    'input': [],
    'output': [],
    'name': '',
}

header = {'Content-Type': 'application/json'}
res = requests.get(HOST_NAME + '/client_request/' + str(CLIENT_ID), headers=header, json=client_data_frame)
print(res)
SUCCESS_FULL_CONNECTION = res.status_code == 200

if TEST_ONNX:
    model_path = "./tests/mobilenetv2-1.0.onnx"
    onnx_model = onnx.load(model_path)
    onnx_json = json.loads(MessageToJson(onnx_model))
    graph = onnx_json['graph']

    weight_payloads = list()
    for x in graph['initializer']:
        tmp = weight_payload.copy()
        if x['dataType'] == 1:
            tmp['weight'] = x['floatData']
        tmp['layer'] = x['name']
        tmp['dims'] = x['dims']
        weight_payloads.append(tmp)

    layer_payloads = list()
    for x in graph['node']:
        tmp = layer_payload.copy()
        tmp['input'] = x['input']
        tmp['output'] = x['output']
        tmp['name'] = x['name']
        if 'attribute' in x.keys():
            tmp['attributes'] = x['attribute']
        tmp['opType'] = x['opType']
        layer_payloads.append(tmp)

    for x in graph['input']:
        pass
    for x in graph['output']:
        pass

    for layer_payload in layer_payloads:
        requests.post(HOST_NAME + '/client_request_worker/push_model/' + str(CLIENT_ID), headers=header, json=layer_payload)
    for weight_payload in weight_payloads:
        requests.post(HOST_NAME + '/client_request_worker/push_weight/' + str(CLIENT_ID), headers=header, json=weight_payload)

    #requests.delete(HOST_NAME + '/client_request/' + str(CLIENT_ID))
