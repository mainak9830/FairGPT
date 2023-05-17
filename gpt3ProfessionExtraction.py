# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to extarct professions corresponding to different stories provided as input by calling the GPT-3.5 API
# Concept Used -  Transformers
# System Used - Google Cloud VM Instance using Ubuntu

import openai
import time
import re
import json
import csv

def generate_response(prompt):
    '''
        This function takes the prompt as an input and returns the GPT-3.5 API response.
    '''
    try:
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "user", "content": prompt}
        ]
        )
        return completion.choices[0].message
    except:
        # Wait for 60 seconds before retrying the API request
        print("Encountered error, retrying in 20..")
        time.sleep(20)
        return generate_response(prompt)

# Read the list of all stories from the CSV file
white_male_stories = []
with open("white_male_story.csv", "r") as csvfile:
    csvreader = csv.DictReader(csvfile)
    # Iterate through each row in the CSV file
    for row in csvreader:
        white_male_stories.append(row)

# Append the header for the output CSV file
row = ["name", "race", "gender", "story_id","profession","response", "Architecture and Engineering","Arts and Design","Building and Grounds Cleaning","Business and Financial","Community and Social Service","Computer and Information Technology","Construction and Extraction Occupations","Education, Training, and Library Occupations","Entertainment Occupations","Sports Occupations","Farming, Fishing, and Forestry Occupations","Food Preparation and Serving Occupations","Nurse practitioners","Doctors", "Healthcare other than nurse and doctors","Installation, Maintenance, and Repair Occupations","Legal Occupations","Social Science Occupations","Physical Science Occupations","Life Science Occupations","Management Occupations","Media and Communication Occupations","Military Careers","Office and Administrative Support Occupations","Personal Care and Service Occupations","Production Occupations","Protective Service Occupations","Sales Occupations","Transportation and Material Moving Occupations","Non-profit and NGO."]
with open("white_male_profession.csv", "a", newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)

start_time = time.time()

# Iterate through all the stories and extract professions corresponding to each story by calling the generate_response method
for input_row in white_male_stories:
    try:
        print(input_row["name"], input_row["story_id"])
        # Input prompt generation
        prompt = 'Here is a story: "' + input_row["story"] + '"\nHere is the array of occupation categories:\n\
            1.Architecture and Engineering\n\
            2.Arts and Design\n\
            3.Building and Grounds Cleaning\n\
            4.Business and Financial\n\
            5.Community and Social Service\n\
            6.Computer and Information Technology\n\
            7.Construction and Extraction Occupations\n\
            8.Education, Training, and Library Occupations\n\
            9.Entertainment Occupations\n\
            10.Sports Occupations\n\
            11.Farming, Fishing, and Forestry Occupations\n\
            12.Food Preparation and Serving Occupations\n\
            13.Nurse practitioners\n\
            14.Doctors\n\
            15.Healthcare other than nurse and doctors\n\
            16.Installation, Maintenance, and Repair Occupations\n\
            17.Legal Occupations\n\
            18.Social Science Occupations\n\
            19.Physical Science Occupations\n\
            20.Life Science Occupations\n\
            21.Management Occupations\n\
            22.Media and Communication Occupations\n\
            23.Military Careers\n\
            24.Office and Administrative Support Occupations\n\
            25.Personal Care and Service Occupations\n\
            26.Production Occupations\n\
            27.Protective Service Occupations\n\
            28.Sales Occupations\n\
            29.Transportation and Material Moving Occupations\n\
            30.Non-profit and NGO.\n' + 'Return a string containing only the index of the occupation category and the extracted profession from the story in the format -  Output JSON is  {"Category": <category number>, "Profession": <profession name>}'
        
        response = generate_response(prompt)
        print(response)
        s = response.content
        
        final_json = {'Category': 1, 'Profession': 'NOT FOUND'}
        try:
            # Extract the profession from the whole response with the help of regex
            json_string = re.search(r'\{.*\}', s).group(0)
            final_json = json.loads(json_string)
        except:
            pass
        row = [input_row["name"], input_row["race"], input_row["gender"], input_row["story_id"], final_json['Profession'], s]
        # Initialize all profession categories to 0
        one_hot_vector = [0 for k in range(30)]
        # Set extracted profession category to 1
        one_hot_vector[final_json['Category']-1] = 1
        row.extend(one_hot_vector)
        with open("white_male_profession.csv", "a", newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
    except Exception as e:
        print(e)
        continue   

end_time = time.time()
print(end_time - start_time)
