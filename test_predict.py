"""
Unit Tests for Income Prediction

author: Yan-ting Liau
date: October 16, 2021
"""
import pandas as pd
from app.predict import Income, incomemodel

from fastapi.testclient import TestClient

model = incomemodel()

from app.main import app
client = TestClient(app)

def test_preprocessing():

    """
    Test whether the column exists and the data type is correct
    input:
            data: pandas dataframe
    output:
             None
    """
    data, _ = model.preprocessing()
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "fnlwgt": pd.api.types.is_integer_dtype,
        "education_num": pd.api.types.is_integer_dtype,
        "capital_gain": pd.api.types.is_integer_dtype,
        "capital_loss": pd.api.types.is_integer_dtype,
        "hours_per_week": pd.api.types.is_integer_dtype,
        "income": pd.api.types.is_integer_dtype,
    }

    # Check columns
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def test_preprocessing_classes():
    """
    Test whether the classes exist in the column
    input:
            data: pandas dataframe
    output:
             None
    """
    data, _ = model.preprocessing()
    # Check that only the known classes are present
    known_classes = [1, 0]

    assert data["income"].isin(known_classes).all()


def test_twoclasses():
    """
    Test whether two classes in training data
    input:
            data: pandas dataframe
    output:
             None
    """
    data, _ = model.preprocessing()

    assert len(set(data["income"])) > 1


def test_preprocessing_ranges():
    """
    Test whether the range of the columnis correct
    input:
          data: pandas dataframe
    output:
          None
    """
    data, _ = model.preprocessing()
    ranges = {
            "age": (15, 95),
            "education_num": (0, 30),
            "hours_per_week": (0, 100)
            }

    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].dropna().between(minimum, maximum).all()


def test_trainingpred_two_classes():
    """
    Test whether two classes are successfully predicted in training data
    input:
          data: pandas dataframe
            training data
          model.train()
    output:
          None
    """
    data, _ = model.preprocessing()
    indep_variable = data.copy()
    indep_variable.pop("income")
    classes = model.train().predict(indep_variable)
    assert len(set(classes)) > 1


def test_testingpred_two_classes():
    """
    Test whether two classes are successfully predicted in testing data
    input:
          data: pandas dataframe
            testing data
          model.test_val()
    output:
          None
    """
    _, data = model.preprocessing()
    indep_variable = data.copy()
    indep_variable.pop("income")
    classes = model.test_val()[0]
    assert len(set(classes)) > 1

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcom to Income Prediction API"}

def test_income_pred_api(inputincome = { "age": 30, "workclass": "State-gov",
    "fnlwgt": 77516, "education": "Masters", "education_num": 15,
    "marital_status": "Never-married", "occupation": "Prof-specialty",
    "relationship": "Not-in-family", "race": "White", "sex": "Female",
    "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
    "native_country": "United-States", "income": 1 }):
    response = client.post("/income_prediction", json=inputincome)
    assert response.status_code == 200

def test_income_pred_type():
    response = client.get("/income_prediction")
    assert type(response.json()) == type({
        "id": "idid",
        "name": "namename",
        "description": "descriptiondescription",
    })

def test_income_pred_value(inputincome = { "age": 30, "workclass": "State-gov",
    "fnlwgt": 77516, "education": "Masters", "education_num": 15,
    "marital_status": "Never-married", "occupation": "Prof-specialty",
    "relationship": "Not-in-family", "race": "White", "sex": "Female",
    "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
    "native_country": "United-States", "income": 1 }):
    response = client.post("/income_prediction", json=inputincome)
    assert response.json()['prediction'] == 0
