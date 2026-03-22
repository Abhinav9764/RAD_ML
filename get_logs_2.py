import requests, json, time

auth_res = requests.post('http://localhost:5001/api/auth/login', json={'username': 'Abhinav', 'password': 'password'})
token = auth_res.json().get('token')

run_res = requests.post('http://localhost:5001/api/pipeline/run', headers={'Authorization': 'Bearer ' + token}, json={'prompt': 'test dummy'})
job_id = run_res.json().get('job_id')
print("Started job:", job_id)

hist_res = requests.get('http://localhost:5001/api/history', headers={'Authorization': 'Bearer ' + token})
print("History immediately:", json.dumps(hist_res.json(), indent=2))

time.sleep(1)

# try again
hist_res2 = requests.get('http://localhost:5001/api/history', headers={'Authorization': 'Bearer ' + token})
print("History after 1s:", json.dumps(hist_res2.json(), indent=2))
