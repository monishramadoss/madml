from flask import Flask, request, jsonify
import pymongo
import gridfs
from bson.objectid import ObjectId

import json

app = Flask(__name__)
app.config["DEBUG"] = True
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
print(myclient.list_database_names())
mydb = myclient["madml"]
worker_db = mydb['workers']
client_db = mydb['clients']
fs = gridfs.GridFS(myclient.weights)

client_data_frame = {
    "progress": "",
    "worker_id": -1,
    'weights': [],
    'layers': [],
}

worker_data_frame = {
    "assembly": "",
    "os": "",
    "cpu_info": "",
    "cpu_cores": "",
    "memory": ""
}


@app.route('/', methods=['GET'])
def home():
    return "<h1>MadMLService</h1><p>Welcome to madml for your AI needs</p>"


@app.route('/client_request_worker/push_weight/<user_id>', methods=['POST'])
def weight_push(user_id):
    content = request.get_json(silent=True)
    if client_db.find_one({'user_id': user_id}):
        json_str = json.dumps(content['weight']).encode('utf-8')
        a = fs.put(json_str)
        content['weight'] = a
        client_db.update({'user_id': user_id}, {'$push': {'weights': content}})
    return ""


@app.route('/client_request_worker/push_model/<user_id>', methods=['POST'])
def model_push(user_id):
    content = request.get_json(silent=True)
    one = client_db.find_one({'user_id': user_id})
    if one:
        client_db.update({'user_id': user_id}, {'$push': {'layers': content}})
    return ""


@app.route('/client_request/<user_id>', methods=['GET', 'DELETE'])
def client_request_worker(user_id):
    if request.method == 'GET':
        data = request.get_json(silent=True)
        data['user_id'] = user_id
        data['weights'] = []
        data['layers'] = []
        if not client_db.find_one({'user_id': user_id}):
            _id = client_db.insert_one(data)
            return _id
        else:
            client_db.update_one({'user_id': user_id}, data)
            return "<h1>MadMLService</h1><p>Already Working</p>"
    elif request.method == 'DELETE':
        client = client_db.find_one({'user_id': user_id})
        if client:
            for weight in client['weights']:
                fs.delete(weight['weight'])
        client_db.delete_one(client)
        return ""

    return "<h1>MadMLService</h1><p>Processing your request</p>"


@app.route('/worker_request/get_weight/<weight_id>', methods=['GET'])
def get_weight(weight_id):
    out = fs.get(ObjectId(weight_id)).read()
    if out:
        out = out.decode('utf-8').replace('[', '').replace(']', '').split(',')
        return jsonify({"weight": out})
    else:
        return jsonify({"weight": []})


@app.route('/worker_request/pull_weight/<client_id>', methods=['GET'])
def get_weights(client_id):
    tensors = client_db.find_one({'_id': ObjectId(client_id)})
    if tensors:
        for weight in tensors['weights']:
            weight['weight'] = str(weight['weight'])
        return jsonify({"tensor_names": tensors['weights']})
    else:
        return jsonify({"tensor_names": []})


@app.route('/worker_request/pull_layers/<user_id>', methods=['GET'])
def get_layers(user_id):
    work = client_db.find_one({"worker_id": "-1"})
    if work:
        client_db.update_one(work, {'$set': {"worker_id": user_id}})
        return jsonify({"layers": work['layers'], 'client_id': str(work['_id'])})
    return jsonify({"layers": [], "client_id": "-1"})


@app.route('/worker_request/<worker_id>', methods=['GET', 'DELETE'])
def ready_worker(worker_id):
    if request.method == 'GET':
        tmp = worker_data_frame.copy()
        tmp.update(request.get_json(silent=True))
        _id = worker_db.insert_one(tmp).inserted_id
        return str(_id)
    elif request.method == 'DELETE':
        one = worker_db.find_one(ObjectId(worker_id))
        worker_db.delete_one(one)
        # need to delete client processing
    return "<h1>MadMLService</h1><p>Processed your request</p>"


app.run(host=None, port=5000)
