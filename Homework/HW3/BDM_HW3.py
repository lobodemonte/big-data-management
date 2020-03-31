import datetime as dt
import numpy as np 
import pandas as pd
import argparse
from pathlib import Path
from pyspark import SparkContext
from pyspark.sql.session import SparkSession

def toCSVLine(entry):
    product = entry[0][0]
    if ',' in entry[0][0]:
        product = '"' + entry[0][0] + '"'
    return product + "," + str(entry[0][1]) + "," + str(entry[1][0]) + "," + str(entry[1][1]) + "," + str(entry[1][2])

def extract_complaints(partId, list_of_records):
    if partId==0: 
        next(list_of_records) # skipping the header line
    import csv
    reader = csv.reader(list_of_records)
    for row in reader:
        try:
            year_received = dt.datetime.strptime(row[0], '%Y-%m-%d').year
            product = row[1].lower()
            company = row[7].lower()            
            yield ((product, year_received, company), 1)
        except:
            print("Exception Occurred", row)

def run_spark(complaints_file_path, output_path):
    sc = SparkContext()
    spark = SparkSession(sc)

    print("Reading input file:", complaints_file_path)
    complaint_info = sc.textFile(complaints_file_path, use_unicode=True).cache()
    print("Number of partitions: ", complaint_info.getNumPartitions())

    complaints = complaint_info.mapPartitionsWithIndex(extract_complaints)

    results = complaint_info.mapPartitionsWithIndex(extract_complaints) \
        .reduceByKey(lambda x,y: x+y) \
        .map(lambda x: ( (x[0][0],x[0][1]), [x[1]] ) ) \
        .reduceByKey(lambda x,y: x+y) \
        .map(lambda x: ( x[0], ( sum(x[1]), len(x[1]), round(max(x[1])*100/sum(x[1])) ) ) ) \
        .sortByKey() \
        .map(toCSVLine) \
    
    print("Saving to: ", output_path)
    results.saveAsTextFile(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_path", type=Path)

    p = parser.parse_args()
    run_spark(str(p.input_file), str(p.output_path))
