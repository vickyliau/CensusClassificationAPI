"""
Prediction Functions for Income

author: Yan-ting Liau
date: October 16, 2021
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, fbeta_score, recall_score, precision_score
from sklearn.preprocessing import OneHotEncoder
from pydantic import BaseModel

DATAPATH = "./app/data/adult.data"
TESTPATH = "./app/data/adult.test"


class Income(BaseModel):
    """
    Constrain data types in model
    input:
        all data types
    output:
        None
    """
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    income: int


def match_categories(data, data_test):
    """
    Match classes between training and testing datasets
    input:
            data: pandas dataframe
                training data
            data_test: pandas dataframe
                testing data
    output:
            data: pandas dataframe
                training data
            data_test: pandas dataframe
                testing data
    """
    for cols in list(set(data.columns) - set(data_test.columns)):
        data_test[cols] = 0
    for cols in list(set(data_test.columns) - set(data.columns)):
        data[cols] = 0
    return data, data_test


class incomemodel:
    """
    prediction on Income
    Input:
        training data
        testing data
        input from API
    output:
        model outputs
    """
    def __init__(self):
        self.datatab = pd.read_csv(DATAPATH, header=None)
        self.test = pd.read_csv(TESTPATH, skiprows=[0], header=None)
        self.mname = "./app/model/income_classifier.joblib"
        try:
            self.data, self.data_test = self.preprocessing()
            self.pipe = joblib.load(self.mname)
            self.prob, self.dep_testvariable = self.test_val()
            self.evaluation = self.evaluation()
        except Exception:
            self.data, self.data_test = self.preprocessing()
            self.pipe = self.train()
            joblib.dump(self.pipe, self.mname)
            self.prob, self.dep_testvariable = self.test_val()
            self.evaluation = self.evaluation()

    # Read and Clean Data
    def preprocessing(self):
        """
        Pre-processing training and testing datasets
        input:
                data: pandas dataframe
                    training data
                data_test: pandas dataframe
                    testing data
        output:
                data: pandas dataframe
                    training data
                data_test: pandas dataframe
                    testing data
        """
        headers = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ]
        data = self.datatab
        data.columns = headers
        data.to_csv('./app/data/training_clean.csv', index=False)
        data["income"] = data["income"].replace(" <=50K", 0)
        data["income"] = data["income"].replace(" >50K", 1)
        cdf = pd.DataFrame(
            OneHotEncoder()
            .fit_transform(data.select_dtypes(include=[object]))
            .toarray(),
            columns=OneHotEncoder()
            .fit(data.select_dtypes(include=[object]))
            .get_feature_names_out(),
        )
        data = data.select_dtypes(include=[int]).join(cdf, how="outer")
        

        data_test = self.test
        data_test.columns = headers
        data_test.to_csv('./app/data/testing_clean.csv', index=False)
        data_test["income"] = data["income"].replace(" <=50K", 0)
        data_test["income"] = data["income"].replace(" >50K", 1)
        cdft = pd.DataFrame(
            OneHotEncoder()
            .fit_transform(data_test.select_dtypes(include=[object]))
            .toarray(),
            columns=OneHotEncoder()
            .fit(data_test.select_dtypes(include=[object]))
            .get_feature_names_out(),
        )
        data_test = data_test.select_dtypes(include=[int])
        data_test = data_test.join(cdft, how="outer")
        
        data, data_test = match_categories(data, data_test)
        data.to_csv('./app/data/training_hotencoding.csv', index=False)
        data_test.to_csv('./app/data/testing_hotencoding.csv', index=False)
        return data, data_test

    def train(self):
        """
        Training the model
        input:
                data: pandas dataframe
                    training data
        output:
                pipe: sklearn model output

        """
        indep_variable = self.data.copy()
        dep_variable = indep_variable.pop("income")
        pipe = RandomForestClassifier(max_depth=2, random_state=0)
        pipe.fit(indep_variable, dep_variable)
        predtab=self.data
        predtab['pred']=pipe.predict(indep_variable)
        predtab.to_csv('./app/data/training_pred.csv',index=False)
        # joblib.dump(pipe, 'income_classifier.joblib')
        return pipe

    def test_val(self):
        """
        Testing the model
        input:
                pipe: sklearn model output

                data_test: pandas dataframe
                    testing data
        output:
                pred: numpy array
                    prediction results
                dep_testvariable: pandas series
                    dependent variable of test data
        """
        indep_testvariable = self.data_test.copy()
        dep_testvariable = indep_testvariable.pop("income")
        pred = self.pipe.predict(indep_testvariable)
        predtab=indep_testvariable
        predtab['pred']=pred
        predtab.to_csv('./app/data/testing_pred.csv',index=False)
        return pred, dep_testvariable

    def slice_performance(self):
        """
        function that computes the performance metrics when the value 
        of a given feature is held fixed. E.g. for education, it would 
        print out the model metrics for each slice of data that has a 
        particular value for education. You should have one set of outputs
        for every single unique value in education.
        input:
                data: pandas dataframe
                    training data
        output:
                pipe: sklearn model output
        """
        indep_variable = self.data.copy()
        sex=[i for i in indep_variable.columns if 'sex' in i]
        for i in range(len(sex)):
            tab = indep_variable[indep_variable[sex[i]]==1]

            dep = tab.pop("income")
            #pipe_sex = RandomForestClassifier(max_depth=2, random_state=0)
            #pipe_sex.fit(tab, dep)
            pred_sex = self.pipe.predict(tab)
            print (sex[i]+': '+str(accuracy_score(dep, pred_sex)))
            np.savetxt('./app/data/'+str(sex[i])+'_slice.txt', pred_sex)


    def evaluation(self):
        """
        Training the model
        input:
                dep_testvariable: pandas series
                    dependent variable of test data
                prob: numpy array
                    prediction results
        output:
                score: integer
                    accuracy of prediction results
        """
        # indep_testvariable = self.data_test.copy()
        # dep_testvariable = indep_testvariable.pop('income')
        accuracy = accuracy_score(self.dep_testvariable, self.prob)
        fbeta = fbeta_score(self.dep_testvariable, self.prob,beta=0.5)
        recall = recall_score(self.dep_testvariable, self.prob)
        precision = precision_score(self.dep_testvariable, self.prob)
        return accuracy, fbeta, recall, precision

    def predict_income(
        self,
        age,
        workclass,
        fnlwgt,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        sex,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country,
        income,
    ):
        """
        Training the model
        input:
                data: pandas dataframe
                    training data
                age: integer
                fnlwgt: integer
                education-num: integer
                capital_gain: integer
                capital_loss: integer
                hours_per_week: integer
                income: integer
                workclass: string
                education: string
                marital_status: string
                occupation: string
                relationship: string
                race: string
                sex: string
                native_country: string_dtype
                income: integer
        output:
                income: integer
                prediction: integer
                probability: integer
                score: integer
        """
        cols = list(self.data.columns)
        cols.remove("income")

        new_data = pd.DataFrame(columns=cols)
        new_data.loc[0] = np.zeros(len(cols))
        new_data["age"] = age
        new_data["fnlwgt"] = fnlwgt
        new_data["education_num"] = education_num
        new_data["capital_gain"] = capital_gain
        new_data["capital_loss"] = capital_loss
        new_data["hours_per_week"] = hours_per_week
        new_data["workclass_ " + workclass] = 1
        new_data["education_ " + education] = 1
        new_data["marital_status_ " + marital_status] = 1
        new_data["occupation_ " + occupation] = 1
        new_data["relationship_ " + relationship] = 1
        new_data["race_ " + race] = 1
        new_data["sex_ " + sex] = 1
        new_data["native_country_ " + native_country] = 1
        new_data = new_data.replace(np.nan, 0)

        prediction = self.pipe.predict(new_data)
        probability = self.pipe.predict_proba(new_data).max()
        val = [[income]]
        score = accuracy_score(val, prediction)
        print([income, int(prediction), probability, score])
        return income, int(prediction), probability, score

if __name__ == "__main__":
    model = incomemodel()
    accuracy, fbeta, recall, precision = model.evaluation
    print ('accuracy'+str(accuracy))
    print ('fbeta'+str(fbeta))
    print ('recall'+str(recall))
    print ('precision'+str(precision))
