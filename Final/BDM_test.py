import pandas as pd
import numpy as np
import traceback
import time

from datetime import datetime

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import col, udf, array, count
from pyspark.sql.functions import broadcast,coalesce, lit
from itertools import chain
from pyspark.sql.functions import col, lit, when, isnull

def to_upper(string):
    if string is None:
        return None
    return string.strip().upper()

def get_county_code(county):
    if county is not None:
        # Boro codes: 1 = MN, 2 = BX, 3 = BK, 4 = QN, 5 = SI
        if county.startswith("M") or county.startswith("N"):
            return 1
        if county in ['BRONX', 'BX', 'PBX']:
            return 2
        if county in ['BK', 'K', 'KING', 'KINGS']:
            return 3
        if county.startswith('Q'):
            return 4
        if county == 'R' or county == 'ST':
            return 5
    return -1

def get_year(string): 
    data_val = datetime.strptime(string.strip(), '%m/%d/%Y')    
    return data_val.year

def get_street_number(street_val):
    if street_val is None:
        return 0
    if type(street_val) is int:
        return street_val
    elems = street_val.split("-")
    new_val = "".join(elems)
    if new_val.isdigit():
        return int(new_val)
    else:
        return 0

def as_digit(val):
    if val:
        return int(val)
    return val

def getOLS(values):
    import statsmodels.api as sm
    X = sm.add_constant(np.arange(len(values)))
    fit = sm.OLS(values, X).fit()
    coef = fit.params[0]
    return float(coef)

def get_violations_df(violations_file, spark):
    get_county_code_udf = udf(get_county_code)
    get_street_number_udf = udf(get_street_number)
    get_year_udf = udf(get_year)
    to_upper_udf = udf(to_upper)
    
    violations_df = spark.read.csv(violations_file, header=True, inferSchema=True)

    violations_df = violations_df.select("Violation County", "House Number", "Street Name", "Issue Date")

    violations_df = violations_df.filter((violations_df['Violation County'].isNotNull()) 
                                         & (violations_df['House Number'].isNotNull()) 
                                         & (violations_df['Street Name'].isNotNull()) 
                                         & (violations_df['Issue Date'].isNotNull())
                                        )

    violations_df = violations_df.withColumn('Violation County', get_county_code_udf(violations_df['Violation County']))
    violations_df = violations_df.withColumn('House Number', get_street_number_udf(violations_df['House Number']))
    violations_df = violations_df.withColumn('Street Name', to_upper_udf(violations_df['Street Name']))
    violations_df = violations_df.withColumn('Issue Date', get_year_udf(violations_df['Issue Date']))

    violations_df = violations_df.withColumnRenamed("Violation County","COUNTY")
    violations_df = violations_df.withColumnRenamed("House Number","HOUSENUM")
    violations_df = violations_df.withColumnRenamed("Street Name","STREETNAME")
    violations_df = violations_df.withColumnRenamed("Issue Date","YEAR")

    violations_df = violations_df.where(violations_df.YEAR.isin(list(range(2015,2020))))
    violations_df = violations_df.repartition(5,'COUNTY')
    violations_df = violations_df.alias('v')
    return violations_df

def get_streets_df(streets_file, spark):
    get_street_number_udf = udf(get_street_number)
    to_upper_udf = udf(to_upper)
    as_digit_udf = udf(as_digit)
    
    streets_df = spark.read.csv(streets_file, header=True, inferSchema=True)

    streets_df = streets_df.select("PHYSICALID","BOROCODE", "FULL_STREE", "ST_LABEL","L_LOW_HN", "L_HIGH_HN", 
                                   "R_LOW_HN", "R_HIGH_HN")

    streets_df = streets_df.withColumn('BOROCODE', as_digit_udf(streets_df['BOROCODE']))
    streets_df = streets_df.withColumn('FULL_STREE', to_upper_udf(streets_df['FULL_STREE']))
    streets_df = streets_df.withColumn('ST_LABEL',   to_upper_udf(streets_df['ST_LABEL']))
    streets_df = streets_df.withColumn('L_LOW_HN',  get_street_number_udf(streets_df['L_LOW_HN']))
    streets_df = streets_df.withColumn('L_HIGH_HN', get_street_number_udf(streets_df['L_HIGH_HN']))
    streets_df = streets_df.withColumn('R_LOW_HN',  get_street_number_udf(streets_df['R_LOW_HN']))
    streets_df = streets_df.withColumn('R_HIGH_HN', get_street_number_udf(streets_df['R_HIGH_HN']))

    streets_df = streets_df.withColumnRenamed("L_LOW_HN","OddLo")
    streets_df = streets_df.withColumnRenamed("L_HIGH_HN","OddHi")
    streets_df = streets_df.withColumnRenamed("R_LOW_HN","EvenLo")
    streets_df = streets_df.withColumnRenamed("R_HIGH_HN","EvenHi")
    
    streets_df = streets_df.repartition(5, 'BOROCODE')
    streets_df = streets_df.alias('s')
    return streets_df

def mapper(row):
    if row['FULL_STREE'] == row['ST_LABEL']:
        yield ( 
                (row['BOROCODE'], row["FULL_STREE"] ), 
                [( row['EvenLo'],row['EvenHi'],row['OddLo'],row['OddHi'], row['PHYSICALID'] )] 
              ) 
    else:
        yield ( 
                (row['BOROCODE'], row["FULL_STREE"]), 
                [( row['EvenLo'],row['EvenHi'],row['OddLo'],row['OddHi'] ,row['PHYSICALID'] )] 
              ) 
        yield ( 
                (row['BOROCODE'], row["ST_LABEL"]), 
                [( row['EvenLo'],row['EvenHi'],row['OddLo'],row['OddHi'], row['PHYSICALID'] ) ]
              ) 

def get_val(borocode, street, housenum):
    val = streets_dict.get( (borocode, street) )
    if val:
        for item in val:
            if int(housenum) % 2 == 0:
                if int(item[0]) >= int(housenum )and int(housenum) <= int(item[1]):
                    return item[4]
            else:
                if int(item[2]) >= int(housenum) and int(housenum) <= int(item[3]):
                    return item[4]      
    return None

def run_spark(output_path):
    sc = SparkContext()
    spark = SparkSession(sc)
    
    streets = "nyc_cscl.csv"
    violations = "nyc_parking_violation/*.csv"
    # streets = "hdfs:///tmp/bdm/nyc_cscl.csv"
    # violations = "hdfs:///tmp/bdm/nyc_parking_violations/"

    violations_df = get_violations_df(violations, spark)
    streets_df = get_streets_df(streets, spark)

    streets_dict = streets_df.rdd.flatMap(mapper).reduceByKey(lambda x,y: x+y).collectAsMap()
    
    get_val_udf = udf(get_val)
    violations_2 = violations_df.withColumn('PHYSICALID', get_val_udf(violations_df['v.County'], 
                                                            violations_df['v.STREETNAME'], violations_df['v.HOUSENUM']
                                                            ))
    violations_2 = violations_2.filter( violations_2['PHYSICALID'].isNotNull() )
    # violations_2.show()
    violations_2 = violations_2.groupBy("PHYSICALID", "YEAR").agg(count("*").alias("YEAR_COUNT"))
    violations_2.createOrReplaceTempView("violations2_results")
    
    summaries = spark.sql(
        "select PHYSICALID, " +
        "MAX(CASE WHEN (YEAR = 2015) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2015, " +
        "MAX(CASE WHEN (YEAR = 2016) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2016, " +
        "MAX(CASE WHEN (YEAR = 2017) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2017, " +
        "MAX(CASE WHEN (YEAR = 2018) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2018, " +
        "MAX(CASE WHEN (YEAR = 2019) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2019  " +
        "from violations2_results " +
        "group by PHYSICALID " +
        "order by PHYSICALID "
    )

    getOLS_udf = udf(getOLS)
    summaries = summaries.withColumn('OLS_COEF', getOLS_udf(array('COUNT_2015', 'COUNT_2016', 'COUNT_2017', 'COUNT_2018', 'COUNT_2019')))
    summaries.write.csv(output_path, header=False)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    
    parser.add_argument("output_path", type=Path)
    p = parser.parse_args()

    print("Output Path: ", str(p.output_path))
    df = run_spark(str(p.output_path))
    # print(df.show(10))
    print("Done")