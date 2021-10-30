# Deploying a Machine Learning Model on Heroku with FastAPI
This project aims for predicting whether income exceeds $50K/yr based on census data >50K, <=50K.




## Set up Environment

### pip install -r /path/to/requirements.txt

## ML Pipeline

The ML pipeline is defined by app.predict.py

## Model Card

### Model Details

The model is built by random forest, defined by max_depth=2, random_state=0
The API is at https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

### Intended use for the model and the intended users

The model is used for predicting income based on census or person data for users who expect to know how to earn >50K.  

### Metrics

#### Overall Performance: accuracy: 0.7599; fbeta: 0.00627; recall: 0.00128; precision: 0.227272

### Data

Census Income Data Set/ https://archive.ics.uci.edu/ml/datasets/census+income

#### Training Data: app/data/adult.data
#### Testing Data: app/data/adult.test
#### Attribute Information

##### age: continuous.
##### workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
##### fnlwgt: continuous.
##### education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
##### education-num: continuous.
##### marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
##### occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
##### relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
##### race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
##### sex: Female, Male.
##### capital-gain: continuous.
##### capital-loss: continuous.
##### hours-per-week: continuous.
##### native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

### Bias inherent either in data or model

Potentially, the model may be biased by sex. 

#### Accuracy of Female Slice: 0.8905
#### Accuracy of Male Slice: 0.7151

## API Instruction using Local Host

API is defined by main.py

### Step1: Type in uvicorn app.main:app --reload in the terminal
### Step2: Go to http://127.0.0.1:8000/docs
### Step3: Default/POST/income_prediction
### Step4: Click Try it out
### Step5: Enter the following dictionary in Request body

{
  "age": 30,
  "workclass": "State-gov",
  "fnlwgt": 77516,
  "education": "Masters",
  "education_num": 15,
  "marital_status": "Never-married",
  "occupation": "Prof-specialty",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Female",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States",
  "income": 1
}

### Step6: Click Execute
### Step7: Got to Server response for results

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/example.png)

## API Continuous Integration using GitHub Actions

https://github.com/features/actions

The CI is defined by config.yaml

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/firstCI.png)

## Data Versioning using Data Version Control

https://dvc.org/

The pipeline is defined by dvc.yaml, and the versioning is stored by dvc.lock

### Deployment if new data or model is available by typing in dvc repro
### Directed Acyclic Graph by typing in dvc dag, defined by dvc.yaml

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/dvcdag.png)
![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/dvclock.png)


## API Continuous Deployment using Heroku

https://dashboard.heroku.com/apps

The app is started by Procfile

The API is deployed at https://census-income-pred.herokuapp.com/

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/continuous_deloyment.png)
![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/screenshot_live_get.png)
![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/liveAPI.png)

### Live API Testing

import requests

import json


response = requests.get('https://census-income-pred.herokuapp.com/')

print(response.status_code)

print(response.json())


data = { "age": 30, "workclass": "State-gov", "fnlwgt": 77516, "education": "Masters", "education_num": 15, "marital_status": "Never-married", "occupation": "Prof-specialty", "relationship": "Not-in-family", "race": "White", "sex": "Female", "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40, "native_country": "United-States", "income": 1 }

r = requests.post('https://census-income-pred.herokuapp.com/income_prediction', data=json.dumps(data))

print(r.status_code)

print(r.json())


