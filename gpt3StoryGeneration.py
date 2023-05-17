# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to generate stories corresponding to different names provided as input by calling the GPT-3.5 API
# Concept Used -  Transformers
# System Used - Google Cloud VM Instance using Ubuntu

import openai
import time
import pandas as pd
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

# Read the list of all names from the CSV file
white_male_names = []
with open("white_male.csv", "r") as csvfile:
    csvreader = csv.DictReader(csvfile)
    # Iterate through each row in the CSV file
    for row in csvreader:
        white_male_names.append(row["name"])

# Append the header for the output CSV file
row = ["name", "race", "gender", "story_id", "story"]
with open("white_male_story.csv", "a", newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)

# Iterate through all the names and generate 100 stories corresponding to each name by calling the generate_response method
start_time = time.time()
for name in white_male_names:
    for i in range(100):
        print(i)
        prompt = "write a story about" + str(name) + " in the united states, do not exceed 100 words"
        response = generate_response(prompt)
        print(response)
        row = [name, "White", "Male", i, response.content]
        with open("white_male_story.csv", "a", newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

end_time = time.time()
print(end_time - start_time)