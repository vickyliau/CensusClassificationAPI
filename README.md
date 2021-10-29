# censusclassificationAPI
This project aims for predicting whether income exceeds $50K/yr based on census data >50K, <=50K.


## Data

Census Income Data Set/ https://archive.ics.uci.edu/ml/datasets/census+income

### Training Data: app/data/adult.data
### Testing Data: app/data/adult.test

## Set up Environment

### pip install -r /path/to/requirements.txt

## Attribute Information:

#### age: continuous.
#### workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#### fnlwgt: continuous.
#### education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#### education-num: continuous.
#### marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#### occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#### relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#### race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#### sex: Female, Male.
#### capital-gain: continuous.
#### capital-loss: continuous.
#### hours-per-week: continuous.
#### native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## ML Pipeline

The ML pipeline is defined by predict.py

## API Instruction

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

## Continuous Integration 

The CI is defined by config.yaml

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/firstCI.png)

## Data Versioning

### Deployment if new data or model is available by typing in dvc repro
### Directed Acyclic Graph by typing in dvc dag, defined by dvc.yaml

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/dvcdag.png)
![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/dvclock.png)


## API Deployment

![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/continuous_deloyment.png)
![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/screenshot live_get.png)
![alt text](https://github.com/vickyting0910/censusclassificationAPI/blob/main/images/liveAPI.png)



