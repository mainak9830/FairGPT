# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to take a list of names as input and predict their corresponding race
# System Used - Google Cloud VM Instance using Ubuntu

# Library Author: Suriyan Laohaprapanon, Gaurav Sood and Bashar Naji
# Library Website Link: https://ethnicolr.readthedocs.io/ethnicolr.html

import pandas as pd
import time

# Import ethnicolr library
from ethnicolr import pred_census_ln

startTime = time.time()

# Read the text file containing all the names into a dataframe
df= pd.read_csv('./namefolder/0bf2052e-3467-42fd-a6ee-ad0841fe9306.txt', delimiter=None, header=None, names=['name'], engine='python')

pred_df  = pred_census_ln(df, 'name', 2010)

# Write the extracted race and the corresponding names to a the CSV file
pred_df[["name", "race"]].to_csv('./raceoutput/gender_output.csv', index=False)

print("Total time taken is ", (time.time() - startTime))