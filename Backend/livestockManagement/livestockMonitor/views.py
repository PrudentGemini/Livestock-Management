from django.shortcuts import render
from livestockHealthClassifier import disease_classifier
from livestockHealthClassifier import diseaseDataset
import pandas as pd


# Create your views here.
# this view is responsible for the home page of our web interface


def homePage(request):
    context = {
        'symptoms1': list(diseaseDataset['Symptom 1'].unique()),
        'symptoms2': list(diseaseDataset['Symptom 2'].unique()),
        'symptoms3': list(diseaseDataset['Symptom 3'].unique()),
    }
    return render(request, 'livestockMonitor/homePage.html', context=context)


def healthPrediction(request):
    animal_name = request.POST['animal-name'].lower()
    animal_symptoms1 = request.POST['symptom1'].lower()
    animal_symptoms2 = request.POST['symptom2'].lower()
    animal_symptoms3 = request.POST['symptom3'].lower()
    animal_age = request.POST['age']
    animal_temp = request.POST['temp']

    context = {
        'animal_name': animal_name,
        'symptom1': animal_symptoms1,
        'symptom2': animal_symptoms2,
        'symptom3': animal_symptoms3,
        'age': animal_age,
        'temp': animal_temp,
        'symptoms1': list(diseaseDataset['Symptom 1'].unique()),
        'symptoms2': list(diseaseDataset['Symptom 2'].unique()),
        'symptoms3': list(diseaseDataset['Symptom 3'].unique()),
    }
    data = pd.DataFrame({
        'Animal': [animal_name],
        'Age': [animal_age],
        'Temperature': [animal_temp],
        'Symptom 1': [animal_symptoms1],
        'Symptom 2': [animal_symptoms2],
        'Symptom 3': [animal_symptoms3]
    })
    context['animalDisease'] = disease_classifier.predict(data)[0]
    return render(request, 'livestockMonitor/homePage.html', context=context)
