import requests, json

try:
    auth_res = requests.post('http://localhost:5001/api/auth/login', json={'username': 'Abhinav', 'password': 'password'})
    data = auth_res.json()
    token = data.get('token') or data.get('access_token')
    if not token:
        print('Login failed:', data)
    else:
        hist_res = requests.get('http://localhost:5001/api/history', headers={'Authorization': 'Bearer ' + token})
        print("Status code:", hist_res.status_code)
        try:
            print("Response:", json.dumps(hist_res.json(), indent=2))
        except:
            print("Raw text:", hist_res.text)
except Exception as e:
    print('Script Error:', e)
