from flask import Flask, request, jsonify
import pymongo
app = Flask(__name__)
app.config["DEBUG"] = True
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
print(myclient.list_database_names())
mydb = myclient["madml"]
worker_db = mydb['workers']
client_db = mydb['clients']

client_data_frame = {
    "user_id": -1,
    "progress": "new",
    "worker_id": -1
}
worker_data_frame = {
    "worker_id": -1,
    "assembly": "",
    "os": "",
    "cpu_info": "",
    "cpu_cores": "",
    "memory": ""
}


@app.route('/', methods=['GET'])
def home():
    return "<h1>MadMLService</h1><p>Welcome to madml for your AI needs</p>"


@app.route('/client_request', methods=['GET', 'POST'])
def client_request():
    if request.method == 'GET':
        client_id = {'user_id': request.form['user_id']}
        if len(request.form) >= len(client_data_frame.keys()):
            tmp_dict = dict([(k, request.form[k]) for k in client_data_frame.keys()])
            if client_db.find_one(client_id, tmp_dict) == None:
                client_db.insert_one(tmp_dict)
            else:
                print('found')
                json = request.form.to_dict()
                print(json)
                return "<h1>MadMLService</h1><p>Already Submitted</p>"
    elif request.method == 'POST':
        print(request.form)
        return ""
    return "<h1>MadMLService</h1><p>Processing your request</p>"


@app.route('/ready_worker', methods=['GET', 'POST', 'PUT'])
def ready_worker():
    if request.method == 'POST':
        worker_id = {'worker_id': request.form['worker_id']}
        if len(request.form) == len(client_data_frame.keys()):
            tmp_dict = dict([(k, request.form[k]) for k in worker_data_frame.keys()])
            if worker_db.find_one(worker_id, tmp_dict) == None:
                worker_db.insert_one(tmp_dict)
    elif request.method == 'GET':
        tmp = client_db.find_one({'worker_id': "-1"})
        if tmp:
            return dict([(k, tmp[k]) for k in client_data_frame.keys()])
        else:
            return client_data_frame
    elif request.method == 'PUT':
        if len(request.form) >= len(client_data_frame.keys()):
            tmp_dict = dict([(k, request.form[k]) for k in client_data_frame.keys()])
            print(tmp_dict)
            if client_db.find_one({'user_id': request.form['user_id'], 'worker_id': "-1"}):
                client_db.update({'user_id': request.form['user_id'], 'worker_id': "-1"}, tmp_dict, upsert=True)

    return "<h1>MadMLService</h1><p>Processed your request</p>"


app.run(host=None, port=5000)
