# This model is based on classification of farm animals as either healthy or not
# if the farm animal is unhealthy, a prediction analysis is made to determine the type of disease it has
# thus, this model helps to detect farm animals diseases earlier using multiple factors and features from the farm
# animal rather than manually inspecting them which might be time-consuming and sometimes inefficient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import pandas as pd

# first we need to load and make some preprocessing activities on our dataset
# this involves converting all categorical data into their corresponding encoded format using the
# OneHotEncoder function
diseaseDataset = pd.read_csv('../../../dataset/animal_disease_dataset.csv')
animalHealthConditionDataset = pd.read_csv('../../../dataset/Animal disease dataset 2.csv')


# making some basic preprocessing on the dataset before feeding them into the machine learning model
def make_clean(dataFrame: pd.DataFrame):
    frame = dataFrame
    # dropping any rows that  contains any null values
    return frame.dropna()


# cleaning the two dataset using the above function
diseaseDataset = make_clean(diseaseDataset)
animalHealthConditionDataset = make_clean(animalHealthConditionDataset)


symptoms1 = diseaseDataset['Symptom 1'].unique()
symptoms2 = diseaseDataset['Symptom 2'].unique()
symptoms3 = diseaseDataset['Symptom 3'].unique()


# splitting our datasets into training and testing features which is used to train our model
diseaseTrain, diseaseTest, diseaseTrainLabel, diseaseTestLabel = train_test_split(
    diseaseDataset.loc[:, ['Animal', 'Age', 'Temperature', 'Symptom 1', 'Symptom 2', 'Symptom 3', ]],
    diseaseDataset.loc[:, ['Disease']], train_size=.75, random_state=5
)

# Converting yes or no to zero or one before passing it train test split
healthLabel = animalHealthConditionDataset.iloc[:, -1].map({"Yes": 1, "No": 0})
print(healthLabel[healthLabel.isna()])
healthTrain, healthTest, healthLabelTrain, healthLabelTest = train_test_split(
    animalHealthConditionDataset.iloc[:, :6], healthLabel, train_size=.75, random_state=5
)


# after the splitting of the two datasets that will be used to train the machine, the next step is to encode our
# categorical data into numerical data before passing it into our machine learning model
animalHealthColumnTransformer = make_column_transformer(
    (OneHotEncoder(), ['AnimalName', 'symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']),
    remainder='passthrough'
)
animalDiseaseColumnTransformer = make_column_transformer(
    (OneHotEncoder(), ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3']), remainder="passthrough"
)
#

# # ---------------------------------- END OF PREPROCESSING STAGE --------------------------------
# selecting the corresponding model that will be used for making classification and the one that will be used for
# making prediction of farm animals disease

logreg = LogisticRegression(solver="lbfgs")
naiveClassifier = MultinomialNB(alpha=0.01)
#
# # creating a pipeline of steps to be taken from the encoder to passing the data to the corresponding machine
# # learning algorithm
animalHealthPipeline = make_pipeline(animalHealthColumnTransformer, logreg)
animalDiseasePipeline = make_pipeline(animalDiseaseColumnTransformer, naiveClassifier)

# # after the preprocessing of the data, we can now use the preprocessed data to train our machine learning model
health_predictor = animalHealthPipeline.fit(healthTrain, healthLabelTrain)  # training animal health model
disease_classifier = animalDiseasePipeline.fit(diseaseTrain, diseaseTrainLabel)  # training animal disease predictor model
# if __name__ == 'main':
#     print(healthTest)
#     animalDiseasePipeline.predict(diseaseTest)
#     animalHealthPipeline.predict(healthTest)
