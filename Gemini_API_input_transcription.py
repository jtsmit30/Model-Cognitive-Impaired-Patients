from urllib import response

from google import genai
import os
import numpy as np
import pandas as pd

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

x = 0
print("Welcome to the Gemini Medical Test Suite! Please select a test to run:")
print("1. Free-form Conversation")
print("2. Adaptive Questioning")
print("3. Reporting and Analysis")
print("4. Predictions and Recommendations")
print("5. Import patient report from TXT file")
while x not in [1, 2, 3, 4, 5]:
    try:
        x = int(input("Enter the number corresponding to the test you want to run: "))
        if x not in [1, 2, 3, 4, 5]:
            print("Invalid input. Please enter a number between 1 and 5.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


def free_form_conversation():
    
    while True:
        string = input("Enter a prompt for Gemini: ") + "keep your response concise"
        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=string
        )
        print(response.text)
        print("Do you want to continue the conversation? (yes/no)")
        continue_conversation = input().lower()
        if continue_conversation != "yes":
            break
    pass
def adaptive_questioning():

    string = "Keep your response concise. Using complex reasoning, based on one of the most prevalent criteria that you believe is present in the patient, including memory, attention, function, and visuospatial abilities, ask a single question:\n"
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=string
    )
    print(response.text)
    pass

def reporting_and_analysis():

    string = "Keep your response concise. Summarize the conversation data and provide insight on patient risks in a report format"
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=string
    )
    print(response.text)
    pass

def predictions_and_recommendations():

    string = "Keep your response concise. Based on the conversation data, provide predictions and recommendations for patient care."
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=string
    )
    print(response.text)
    pass

def test_gemini_response_length(string):
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=string
    )
    assert len(response.text) <= 500, "Response exceeds 500 characters"

def import_patient_report_from_txt():
    #upload patient report to a txt file, copy path and read it in to test the model's responses
    with open("C:\\Users\\Jason\\OneDrive\\Desktop\\prompt_example_1.txt", "r") as file:
        Txtfile = file.read()
        
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=Txtfile
    )
    print(response.text)
    pass
if x == 1:
    free_form_conversation()
elif x == 2:
    adaptive_questioning()
elif x == 3:
    reporting_and_analysis()
elif x == 4:
    predictions_and_recommendations()
elif x == 5:
    import_patient_report_from_txt()