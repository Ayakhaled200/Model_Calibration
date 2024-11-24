import pandas as pd
import http.client
import json

port = input("Enter port: ")
endpoint = input("Enter endpoint: ")
name = input("Enter data team member name: ")

conn = http.client.HTTPConnection(port)
HEADERS = {
    "Content-Type": "application/json",
    "odata": "verbose",
    "Host": port
}

for ds in ['train', 'test']:
    data = pd.read_csv(f'Knowledge_base_{ds}.csv')
    target = data['class']
    data.drop('class', axis=1, inplace=True)
    features = data.to_dict(orient='index')
    features_dict = json.loads(json.dumps(features, default=str))

    payload = json.dumps(features_dict)
    print(payload)
    conn.request("POST", endpoint, payload, HEADERS)
    response = conn.getresponse()
    res = response.read()

    if response.status == 200:
        res_dict = json.loads(res.decode('utf-8'))
        print(res_dict)
        preds_df = pd.DataFrame.from_dict(res_dict, orient='index')
        preds_df['class'] = target.values
        preds_df.to_csv(f'{name}_{ds}_predictions.csv', index=False)
    else:
        print(f"Status Code: {response.status}")
        print(f"Reason: {response.reason}")

conn.close()
