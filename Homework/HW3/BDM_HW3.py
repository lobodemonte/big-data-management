import datetime as dt
import numpy as np 
import pandas as pd
import argparse
from pathlib import Path
from pyspark import SparkContext
from pyspark.sql.session import SparkSession

def toCSVLine(entry):
    if ',' in entry[0]:
        entry[0] = '"' + entry[0] + '"'
    return ','.join(str(elem) for elem in entry)

def extract_complaints(partId, list_of_records):
    if partId==0: 
        next(list_of_records) # skipping the header line
    import csv
    reader = csv.reader(list_of_records)
    for row in reader:
        year_received = dt.datetime.strptime(row[0], '%m/%d/%Y').year
        product = row[1].lower()
        company = row[7].lower()            
        yield ((product, year_received, company), 1)

def run_spark(complaints_file_path, output_path):
    sc = SparkContext()
    spark = SparkSession(sc)

    complaint_info = sc.textFile(input_file, use_unicode=True).cache()
    print("Number of partitions: ", complaint_info.getNumPartitions())

    complaints = complaint_info.mapPartitionsWithIndex(extract_complaints)

    temp = complaint_info.mapPartitionsWithIndex(extract_complaints) \
        .reduceByKey(lambda x,y: x+y) \
        .map(lambda x: ( (x[0][0],x[0][1]), [x[1]] ) ) \
        .reduceByKey(lambda x,y: x+y) \
        .map(lambda x: ( x[0], ( sum(x[1]), len(x[1]), round(max(x[1])*100/sum(x[1])) ) ) ) \
        .sortByKey() \
        .map(lambda x: (x[0] + x[1]))

    temp.saveAsTextFile(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_path", type=Path)

    p = parser.parse_args()
    if (p.input_file.exists())):
        run_spark(str(p.input_file), str(p.output_path))
    else:
        print("Input File Path: {}, Exists: {}".format(p.input_file, p.input_file.exists()))
