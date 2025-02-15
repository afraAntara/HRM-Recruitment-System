import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


with open('pr_classifier.pickle','rb') as f:
    clf = pickle.load(f)


def predict(data):
    df = data
    emp_id = df.Employee_ID.tolist()
    df = df.drop(
        ['Attrition', 'MaritalStatus', 'Gender', 'DailyRate', 'Employee_ID', 'StandardHours', 'Over18', 'EmployeeCount',
         'EmployeeNumber', 'HourlyRate'], axis=1)

    newdf = df.values.tolist()

    df = pd.DataFrame(newdf,
                      columns=['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
                               'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                               'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                               'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                               'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                               'YearsSinceLastPromotion', 'YearsWithCurrManager'], index=emp_id)


    all_data = df
    all_data = all_data.drop(['StockOptionLevel', 'MonthlyRate', 'YearsWithCurrManager', 'PerformanceRating'], axis=1)

    all_data['OverTime'] = all_data['OverTime'].map({'No': 0, 'Yes': 1}).to_frame()
    all_data['JobRole'] = all_data['JobRole'].map(
        {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 'Manufacturing Director': 3,
         'Healthcare Representative': 4, 'Manager': 5, 'Sales Representative': 6, 'Research Director': 7,
         'Human Resources': 8}).to_frame()
    all_data['EducationField'] = all_data['EducationField'].map(
        {'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4,
         'Human Resources': 5}).to_frame()
    all_data['Department'] = all_data['Department'].map(
        {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2}).to_frame()
    all_data['BusinessTravel'] = all_data['BusinessTravel'].map(
        {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2}).to_frame()


    scaler = StandardScaler()

    scaler.fit(all_data)
    # transform data
    x = scaler.transform(all_data)



    pred = clf.predict(x)
    PerformanceRating = pred
    data['PerformanceRating']= PerformanceRating
    st.write(data['PerformanceRating'])
    # st.write(pred)


def rank(data):
    df = data
    val_data = df.drop(['StandardHours', 'Over18', 'EmployeeCount', 'EmployeeNumber'], axis=1)
    val_data['OverTime'] = val_data['OverTime'].map({'No': 0, 'Yes': 1}).to_frame()
    val_data['JobRole'] = val_data['JobRole'].map(
        {'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2, 'Manufacturing Director': 3,
         'Healthcare Representative': 4, 'Manager': 5, 'Sales Representative': 6, 'Research Director': 7,
         'Human Resources': 8}).to_frame()
    val_data['EducationField'] = val_data['EducationField'].map(
        {'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4,
         'Human Resources': 5}).to_frame()
    val_data['Department'] = val_data['Department'].map(
        {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2}).to_frame()
    val_data['BusinessTravel'] = val_data['BusinessTravel'].map(
        {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2}).to_frame()
    val_data['Attrition'] = val_data['Attrition'].map({'No': 0, 'Yes': 1}).to_frame()
    val_data['Gender'] = val_data['Gender'].map({'Male': 0, 'Female': 1}).to_frame()
    val_data['MaritalStatus'] = val_data['MaritalStatus'].map({'Single': 1, 'Married': 0}).to_frame()
    df = df.drop(
        ['MaritalStatus', 'DailyRate', 'StandardHours', 'Over18', 'EmployeeCount', 'EmployeeNumber', 'HourlyRate'],
        axis=1)
    all_data = df

    #bins
    bins = [0, 5, 10, 15, 20, 25, 100]
    labels = [5, 4, 3, 2, 1, 0]
    all_data['DistanceFromHome'] = pd.cut(all_data['DistanceFromHome'], bins=bins, labels=labels, include_lowest=True)
    bins = [18, 30, 40, 50, 60]
    labels = [4, 3, 2, 1]
    all_data['Age'] = pd.cut(all_data['Age'], bins=bins, labels=labels, include_lowest=True)

    data = df
    data = data.drop(['Employee_ID'], axis=1)
    all_data = all_data.drop(['Employee_ID'], axis=1)
    # encoding string vales and replacing them with encoded values
    all_data = all_data.drop(
        ['WorkLifeBalance', 'Department', 'Gender', 'JobRole', 'YearsAtCompany', 'MonthlyIncome', 'EducationField',
         'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'StockOptionLevel',
         'RelationshipSatisfaction', 'PercentSalaryHike', 'MonthlyRate', 'EnvironmentSatisfaction', 'JobSatisfaction',
         'Attrition', 'BusinessTravel'], axis=1)
    all_data['OverTime'] = all_data['OverTime'].map({'No': 0, 'Yes': 1}).to_frame()

    #MCDM
    #Creating list of values for 10 features
    values = all_data.values.tolist()
    print("This is the values:")

    emp_id = val_data.Employee_ID.tolist()
    emp_data = pd.DataFrame(values, index=emp_id)
    #Normalisation
    emp_data_norm = emp_data / (np.sqrt((np.power(emp_data, 2)).sum(axis=0)))
    print(emp_data_norm)
    #weights
    n = 10
    v = 0.1
    w = [v] * n

    emp_data_norm_w = emp_data_norm * w

    positive_ideal = emp_data_norm_w.max()
    negative_ideal = emp_data_norm_w.min()
    ED_P = np.sqrt((np.power(emp_data_norm_w - positive_ideal, 2)).sum(axis=1))
    ED_N = np.sqrt((np.power(emp_data_norm_w - negative_ideal, 2)).sum(axis=1))
    similarity_score = ED_N / (ED_P + ED_N)
    similarity_score.unique()

    data['Education'] = data['Education'].map(
        {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}).to_frame()
    data['EnvironmentSatisfaction'] = data['EnvironmentSatisfaction'].map(
        {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}).to_frame()
    data['JobInvolvement'] = data['JobInvolvement'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}).to_frame()
    data['JobSatisfaction'] = data['JobSatisfaction'].map({1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}).to_frame()
    data['RelationshipSatisfaction'] = data['RelationshipSatisfaction'].map(
        {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}).to_frame()
    data['PerformanceRating'] = data['PerformanceRating'].map(
        {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}).to_frame()
    data['WorkLifeBalance'] = data['WorkLifeBalance'].map({1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}).to_frame()
    data.head()

    emp_id = val_data.Employee_ID.tolist()

    data['Score'] = np.array(similarity_score)

    newdf = data.values.tolist()
    employees = pd.DataFrame(newdf, columns=['Age', 'Attrition', 'BusinessTravel', 'Department', 'DistanceFromHome',
                                             'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
                                             'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
                                             'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
                                             'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
                                             'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                                             'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                             'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Score'], index=emp_id)

    sorted_scores = employees.sort_values(by=['Score'], ascending=False)



    st.write(sorted_scores)  # displays the table of data


#Title of App
st.title("Ranking using MCDM app")

# Add sidebar
st.sidebar.subheader("Upload File")

# Setup file upload
uploaded_file = st.sidebar.file_uploader(label="Upload CSV file here!",type=['csv'])

global data
if uploaded_file is not None:
    print("hello")
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        data=pd.read_excel(uploaded_file)

try:
    st.write(data)
except Exception as e:
    print(e)
    st.write("Please upload file to application")

if st.button("Predict"):
    predict(data)

if st.button("Rank"):
    rank(data)







