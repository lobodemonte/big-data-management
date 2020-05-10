import pandas as pd
import numpy as np
import traceback

from datetime import datetime

from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import col, udf, array
from pyspark.sql import SQLContext
from pyspark import sql

streets = "nyc_cscl.csv"
violations = "nyc_parking_violations_2015_sample.csv"

# streets = "hdfs:///tmp/bdm/nyc_cscl.csv"
# violations = "hdfs:///tmp/bdm/nyc_parking_violations/"

def to_upper(string):
    if string is None:
        return None
    return string.strip().upper()

def is_digit(value):
    if value:
        return value.isdigit()
    else:
        return False

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

def get_house_number(house_val):
    if house_val is None:
        return None
    if type(house_val) is int:
        return house_val
    elems = house_val.split("-")
    new_val = "".join(elems)
    if new_val.isdigit():
        return int(new_val)
    else:
        return None
    
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
    
def getOLS(values):
    import statsmodels.api as sm
    X = sm.add_constant(np.arange(len(values)))
    fit = sm.OLS(values, X).fit()
    coef = fit.params[0]
    return float(coef)

get_street_number_udf = udf(get_street_number)
get_house_number_udf = udf(get_house_number)
get_county_code_udf = udf(get_county_code)
get_year_udf = udf(get_year)
to_upper_udf = udf(to_upper)
is_digit_udf = udf(is_digit)

def get_violations_df(violations_file, sqlContext):
    violations_df = sqlContext.read.format("csv") \
    .option("delimiter",",") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(violations_file)

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
    return violations_df

def get_streets_df(streets_file, sqlContext):
    streets_df = sqlContext.read.format("csv") \
    .option("delimiter",",") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(streets_file)

    streets_df = streets_df.select("PHYSICALID","BOROCODE", "FULL_STREE", "ST_LABEL","L_LOW_HN", "L_HIGH_HN", 
                                "R_LOW_HN", "R_HIGH_HN")
    streets_df = streets_df.withColumn('FULL_STREE', to_upper_udf(streets_df['FULL_STREE']))
    streets_df = streets_df.withColumn('ST_LABEL',   to_upper_udf(streets_df['ST_LABEL']))
    streets_df = streets_df.withColumn('L_LOW_HN',   get_street_number_udf(streets_df['L_LOW_HN']))
    streets_df = streets_df.withColumn('L_HIGH_HN',   get_street_number_udf(streets_df['L_HIGH_HN']))
    streets_df = streets_df.withColumn('R_LOW_HN',   get_street_number_udf(streets_df['R_LOW_HN']))
    streets_df = streets_df.withColumn('R_HIGH_HN',   get_street_number_udf(streets_df['R_HIGH_HN']))

    streets_df = streets_df.withColumnRenamed("L_LOW_HN","OddLo")
    streets_df = streets_df.withColumnRenamed("L_HIGH_HN","OddHi")
    streets_df = streets_df.withColumnRenamed("R_LOW_HN","EvenLo")
    streets_df = streets_df.withColumnRenamed("R_HIGH_HN","EvenHi") 
    return streets_df   

def run_spark(output_path):
    sc = SparkContext()
    sqlContext = sql.SQLContext(sc)

    streets_df = get_streets_df(streets, sqlContext)
    violations_df = get_violations_df(violations, sqlContext)

    streets_df.registerTempTable("streets")
    violations_df.registerTempTable("violations")

    merged_df = sqlContext.sql(
        "select s.PHYSICALID, v.YEAR " +
        "from streets s left join violations v " +
        "on s.BOROCODE = v.County " +
        "and ( s.FULL_STREE = v.STREETNAME or s.ST_LABEL = v.STREETNAME ) " +
        "and ( (v.HOUSENUM % 2 = 0 and v.HOUSENUM between s.EvenLo and s.EvenHi) or " +
            "(v.HOUSENUM % 2 = 1 and v.HOUSENUM between s.OddLo and s.OddHi) ) " 
    )
    merged_df.registerTempTable("merged_results")

    merged_df = sqlContext.sql(
        "select m.PHYSICALID, m.YEAR, count(m.YEAR) as YEAR_COUNT " +
        "from merged_results m  " +
        "group by m.PHYSICALID, m.YEAR"
    )
    merged_df.registerTempTable("merged_results")

    summaries = sqlContext.sql(
        "select m.PHYSICALID, " +
        "MAX(CASE WHEN (YEAR = 2015) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2015, " +
        "MAX(CASE WHEN (YEAR = 2016) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2016, " +
        "MAX(CASE WHEN (YEAR = 2017) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2017, " +
        "MAX(CASE WHEN (YEAR = 2018) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2018, " +
        "MAX(CASE WHEN (YEAR = 2019) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2019  " +
        "from merged_results m  " +
        "group by m.PHYSICALID " +
        "order by m.PHYSICALID "
    )

    getOLS_udf = udf(getOLS)

    summaries = summaries.withColumn('OLS_COEF', 
                    getOLS_udf(array('COUNT_2015', 'COUNT_2016', 'COUNT_2017', 'COUNT_2018', 'COUNT_2019')))

    summaries.write.csv(output_path, header=False)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    
    parser.add_argument("output_path", type=Path)
    p = parser.parse_args()

    print("Output Path: ", str(p.output_path))
    run_spark(str(p.output_path))
    print("Done")
