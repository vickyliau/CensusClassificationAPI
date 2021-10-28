import requests
import json

response = requests.post('https://census-income-pred.herokuapp.com/')
r = requests.post('https://census-income-pred.herokuapp.com/income_prediction')

print(response.status_code)
print(response.json())

