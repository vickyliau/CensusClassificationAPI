import requests
import json

response = requests.get('https://census-income-pred.herokuapp.com/')
print(response.status_code)
print(response.json())

r = requests.post('https://census-income-pred.herokuapp.com/income_prediction')
data = { "age": 30, "workclass": "State-gov", "fnlwgt": 77516, "education": "Masters", "education_num": 15, "marital_status": "Never-married", "occupation": "Prof-specialty", "relationship": "Not-in-family", "race": "White", "sex": "Female", "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States", "income": 1 }
r = requests.post('https://census-income-pred.herokuapp.com/income_prediction', data=json.dumps(data))
print(r.status_code)
print(r.json())


