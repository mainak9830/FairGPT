# Team Members - Aishik Deb, Mainak Adak, Hao Lin, Yuqing Wang
# Description - Python file to extract names from Wikipedia JSON dataset using pre-trained BERT model and Pyspark
# Data FrameWork Used -  PySpark
# Concept Used - Transformers
# System Used - Google Cloud DataProc Cluster running Ubuntu

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
import re
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, collect_list
from pyspark.sql.types import ArrayType, StringType


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def include(name):
    """
     Function to discard names that has any character other than alphabet and length of each parts of name less than 4
    """
    for n in name:
        if(len(n) < 4 or re.search('[^a-z^A-Z]', n)):
            return False
    
    return True

def preprocess(d):
    """
        Function to separate out the names using B-PER and I-PER tag of tokens
    """
    
    ner_results = nlp(d)

    name = []
    names = []
    for result in ner_results:
        
        if(result['entity'] == 'B-PER'):
            if(len(name) > 1 and include(name)):
                names.append(' '.join(name))
                
                # of.write(' '.join(name)+'\n')
            name = []
            name.append(result['word'])
        if(result['entity'] == 'I-PER'):
            name.append(result['word'])
            


    if(len(name) > 1 and include(name)):
        names.append(' '.join(name))
        # of.write(' '.join(name)+'\n')
    if(len(names)):
        for name in names:
            print(name)
    return names

 
# Iterating through all the files of a directory
for file in os.listdir('dataset'):
    
    # if(file != "sample.json"):
    #     continue
    print(file)
    
    spark = SparkSession.builder.appName("JSON Reader").config("spark.sql.broadcastTimeout", "3600").getOrCreate()
    
    #read the json file
    json_df = spark.read.option("maxRecordsInMemory", "100000").option("maxRecordsPerFile", "200000").option("multiline", "true").json(os.path.join('dataset',file))
    

    preprocess_udf = udf(preprocess, ArrayType(StringType()))
    
    #preprocess the text part of json and generate the list of names seen
    new_df = json_df.withColumn("preprocessed_field", preprocess_udf(col("text")))
    print(json_df.printSchema())
    
    
    # Collect the lists
    combined_list = new_df.select("preprocessed_field"). \
                    rdd.flatMap(lambda x: x).collect()
    print(new_df.collect())
    
    # writing the names to a file
    oname = file.split('.')[0] + '.txt'
    of = open(os.path.join('output',oname), 'a')
    for name_list in combined_list:
        for name in name_list:
            of.write(name+"\n")
    of.close()

 
    
   

    
    
