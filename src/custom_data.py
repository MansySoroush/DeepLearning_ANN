import pandas as pd

class CustomData:
    def __init__(  self,
        credit_score: int,
        geography: str,
        gender: str,
        age: int,
        tenure: int,
        balance: float,
        number_of_products: int,
        has_cr_card: int,
        is_active_member: int,
        estimated_salary: float,
        exited:int,
        is_classifier: bool):

        self.credit_score = credit_score
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.number_of_products = number_of_products
        self.has_cr_card = has_cr_card
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary
        self.exited = exited
        self.is_classifier = is_classifier

    def get_data_as_data_frame(self):
        if self.is_classifier:
            custom_data_input_dict = {
                "CreditScore": [self.credit_score],
                "Geography": [self.geography],
                "Gender": [self.gender],
                "Age": [self.age],
                "Tenure": [self.tenure],
                "Balance": [self.balance],
                "NumOfProducts": [self.number_of_products],
                "HasCrCard": [self.has_cr_card],
                "IsActiveMember": [self.is_active_member],
                "EstimatedSalary": [self.estimated_salary]
            }
        else:
            custom_data_input_dict = {
                "CreditScore": [self.credit_score],
                "Geography": [self.geography],
                "Gender": [self.gender],
                "Age": [self.age],
                "Tenure": [self.tenure],
                "Balance": [self.balance],
                "NumOfProducts": [self.number_of_products],
                "HasCrCard": [self.has_cr_card],
                "IsActiveMember": [self.is_active_member],
                "Exited": [self.exited]
            }

        return pd.DataFrame(custom_data_input_dict)
        
    def __str__(self):
        if self.is_classifier:
            return f"CreditScore={self.credit_score}, Geography = {self.geography}, Gender = {self.gender},\nAge = {self.age}, Tenure = {self.tenure}, Balance = {self.balance},\nNumOfProducts = {self.number_of_products}, HasCrCard = {self.has_cr_card}, IsActiveMember = {self.is_active_member},\nEstimatedSalary = {self.estimated_salary})"
        else:
            return f"CreditScore={self.credit_score}, Geography = {self.geography}, Gender = {self.gender},\nAge = {self.age}, Tenure = {self.tenure}, Balance = {self.balance},\nNumOfProducts = {self.number_of_products}, HasCrCard = {self.has_cr_card}, IsActiveMember = {self.is_active_member},\Exited = {self.exited})"

