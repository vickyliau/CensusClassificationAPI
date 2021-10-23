"""
Settings of FastAPI

author: Yan-ting Liau
date: October 16, 2021
"""

from fastapi import FastAPI
import uvicorn
from predict import Income, incomemodel

app = FastAPI()

model = incomemodel()


@app.get("/")
async def read_root():
    """
    Start Message for API
    input:
            None
    output:
            None
    """
    return {"message": "Welcom to Income Prediction API"}


@app.post("/income_prediction")
async def income_pred(inputincome: Income):
    """
    Inputs for API prediction
    input:
        JSON file with age, workclass, fnlwgt, education, education_num,
        marital_status, occupation, relationship, race, sex, capital_gain,
        capital_loss, hours_per_week, native_country, income
    output:
            income
            prediction
            probability
            score
    """
    data = inputincome.dict()
    income, prediction, probability, score = model.predict_income(
        data["age"],
        data["workclass"],
        data["fnlwgt"],
        data["education"],
        data["education_num"],
        data["marital_status"],
        data["occupation"],
        data["relationship"],
        data["race"],
        data["sex"],
        data["capital_gain"],
        data["capital_loss"],
        data["hours_per_week"],
        data["native_country"],
        data["income"],
    )

    return {
        "income": income,
        "prediction": prediction,
        "probability": probability,
        "score": score,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
