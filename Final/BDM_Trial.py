import pandas as pd
import numpy as np
import string
import traceback
import time

from datetime import datetime
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array, count, split
from pyspark.sql.functions import broadcast, coalesce, lit

streets = "nyc_cscl.csv"
violations = "nyc_parking_violation/*.csv"
#streets = "hdfs:///tmp/bdm/nyc_cscl.csv"
#violations = "hdfs:///tmp/bdm/nyc_parking_violation/*.csv"

def to_upper(raw_str):
    import string
    if raw_str is not None and raw_str != "":
        clean_str = raw_str.strip(string.punctuation)
        return clean_str.strip().upper()
    return None

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
        if county.startswith('R') or county.startswith('ST'):
            return 5
    return None

def get_year(date_string):
    if date_string is not None:
        data_val = datetime.strptime(date_string.strip(), '%m/%d/%Y')  
        if data_val.year in list(range(2015,2020)):
            return data_val.year
    return None

def get_street_number(street_val_raw, default=None):
    if street_val_raw is not None:
        if type(street_val_raw) is int:
            return street_val_raw
        
        street_val = street_val_raw.strip(string.ascii_letters)
        elems = street_val.split("-")  
        
        if len(elems) == 1 and elems[0].isdigit():
            return int(elems[0])
        
        elif len(elems) == 2 and elems[0].isdigit() and elems[1].isdigit():
            new_val = elems[0] + "{:01d}".format(int(elems[1]))
            if new_val.isdigit():
                return int(new_val)   
        
        elif len(elems) == 3 and elems[0].isdigit() and elems[2].isdigit():
            new_val = elems[0] + "{:01d}".format(int(elems[2]))
            if new_val.isdigit():
                return int(new_val) 
        else:
            new_val = "".join(elems)
            if new_val.isdigit():
                return int(new_val)
    return default

def as_digit(val):
    if val:
        return int(val)
    return val

def getOLS(values):
    import statsmodels.api as sm
    X = sm.add_constant([1,2,3,4,5])
    fit = sm.OLS(values, X).fit()
    coef = fit.params[1]
    return float(coef)

def get_violations_df(violations_file, spark):
    get_county_code_udf = udf(get_county_code)
    get_street_number_udf = udf(get_street_number)
    get_year_udf = udf(get_year)
    to_upper_udf = udf(to_upper)
    
    violations_df = spark.read.csv(violations_file, header=True, inferSchema=False)

    violations_df = violations_df.select("Violation County", "House Number", "Street Name", "Issue Date")

    violations_df = violations_df.withColumnRenamed("Violation County","COUNTY")
    violations_df = violations_df.withColumnRenamed("House Number","HOUSENUM")
    violations_df = violations_df.withColumnRenamed("Street Name","STREETNAME")
    violations_df = violations_df.withColumnRenamed("Issue Date","YEAR")
    
    violations_df = violations_df.withColumn('HOUSENUM_RAW', col("HOUSENUM"))

    split_col = split(violations_df['HOUSENUM'], '-')
    violations_df = violations_df.withColumn('NUM0', split_col.getItem(0))
    violations_df = violations_df.withColumn('NUM1', split_col.getItem(1))
    violations_df = violations_df.withColumn('NUM0', get_street_number_udf(violations_df['NUM0'],lit(0)))
    violations_df = violations_df.withColumn('NUM1', get_street_number_udf(violations_df['NUM1'],lit(0)))

    
    violations_df = violations_df.withColumn('COUNTY', get_county_code_udf(violations_df['COUNTY']))
    
    violations_df = violations_df.withColumn('HOUSENUM', get_street_number_udf(violations_df['HOUSENUM']))
    
    violations_df = violations_df.withColumn('STREETNAME', to_upper_udf(violations_df['STREETNAME']))
    violations_df = violations_df.withColumn('YEAR', get_year_udf(violations_df['YEAR']))

    violations_df = violations_df.filter((violations_df['COUNTY'].isNotNull()) 
                                         & (violations_df['HOUSENUM'].isNotNull()) 
                                         & (violations_df['STREETNAME'].isNotNull()) 
                                         & (violations_df['YEAR'].isNotNull())
                                        )
    
    violations_df = violations_df.withColumn("COUNTY", violations_df["COUNTY"].cast("integer"))
    violations_df = violations_df.withColumn("HOUSENUM", violations_df["HOUSENUM"].cast("integer"))
    violations_df = violations_df.withColumn("YEAR", violations_df["YEAR"].cast("integer"))
    
    violations_df = violations_df.withColumn('NUM0', violations_df["NUM0"].cast("integer"))
    violations_df = violations_df.withColumn('NUM1', violations_df["NUM1"].cast("integer"))

    violations_df = violations_df.repartition(5,'COUNTY')
    violations_df = violations_df.alias('v')
    return violations_df

def get_streets_df(streets_file, spark):
    get_street_number_udf = udf(get_street_number)
    to_upper_udf = udf(to_upper)
    as_digit_udf = udf(as_digit)
    
    streets_df = spark.read.csv(streets_file, header=True, inferSchema=False)

    streets_df = streets_df.select("PHYSICALID","BOROCODE", "FULL_STREE", "ST_LABEL","L_LOW_HN", "L_HIGH_HN", 
                                   "R_LOW_HN", "R_HIGH_HN")

    streets_df = streets_df.withColumnRenamed("L_LOW_HN","OddLo")
    streets_df = streets_df.withColumnRenamed("L_HIGH_HN","OddHi")
    streets_df = streets_df.withColumnRenamed("R_LOW_HN","EvenLo")
    streets_df = streets_df.withColumnRenamed("R_HIGH_HN","EvenHi")
    
    streets_df = streets_df.filter((streets_df['BOROCODE'].isNotNull()) 
                                   & (streets_df['PHYSICALID'].isNotNull())
                                  )
    
    streets_df = streets_df.withColumn('BOROCODE', as_digit_udf(streets_df['BOROCODE']))
    streets_df = streets_df.withColumn('FULL_STREE', to_upper_udf(streets_df['FULL_STREE']))
    streets_df = streets_df.withColumn('ST_LABEL',   to_upper_udf(streets_df['ST_LABEL']))

    streets_df = streets_df.withColumn("BOROCODE", streets_df["BOROCODE"].cast("integer"))
    streets_df = streets_df.repartition(5, 'BOROCODE')
    streets_df = streets_df.alias('s')
    return streets_df

def process_num(str_num, default=None):
    if str_num is not None:

        str_num_clean = str_num.strip(string.ascii_letters)
        elems = str_num_clean.split("-")  
        
        if len(elems) == 1 and elems[0].isdigit():
            return (0, int(elems[0]))
        
        elif len(elems) == 2 and elems[0].isdigit() and elems[1].isdigit():
             return (int(elems[0]), int(elems[1]))
        
        elif len(elems) == 3 and elems[0].isdigit() and elems[2].isdigit():
             return (int(elems[0]), int(elems[2]))
        else:
            new_val = "".join(elems)
            if new_val.isdigit():
                return (0,int(new_val))
    return (0,0)

def mapper_2(row):
    evenLo = process_num(row['EvenLo'])
    evenHi = process_num(row['EvenHi'])
    oddLo = process_num(row['OddLo'])
    oddHi = process_num(row['OddHi'])
    
    if row['FULL_STREE'] == row['ST_LABEL']:
        yield ( 
                (row['BOROCODE'], row["FULL_STREE"] ), 
                [( evenLo, evenHi, oddLo, oddHi, row['PHYSICALID'] )] 
              ) 
    else:
        yield ( 
                (row['BOROCODE'], row["FULL_STREE"]), 
                [( evenLo, evenHi, oddLo, oddHi, row['PHYSICALID'] )] 
              ) 
        yield ( 
                (row['BOROCODE'], row["ST_LABEL"]), 
                [( evenLo, evenHi, oddLo, oddHi, row['PHYSICALID'] ) ]
              )  

def search_candidates_2(candidates, housenum):
    for item in candidates:
        if housenum[1] % 2 == 0:
            if item[0] <= housenum and housenum <= item[1]:
                return item[4]
        else:
            if item[2] <= housenum and housenum <= item[3]:
                return item[4]  
    return None
    
def run_spark(output_file):
    sc = SparkContext()
    spark = SparkSession(sc)

    violations_df = get_violations_df(violations, spark)
    streets_df = get_streets_df(streets, spark)

    streets_dict = streets_df.rdd.flatMap(mapper_2).reduceByKey(lambda x,y: x+y).collectAsMap()
    streets_dict_bc = sc.broadcast(streets_dict)

    def get_val_2(borocode, street, num0, num1):
        res = None
        housenum = (num0, num1)
        if num0 != 0 and num1 == 0:
            housenum = (num1, num0)
        candidates = streets_dict_bc.value.get( (borocode, street) )
        
        if candidates:
            res = search_candidates_2(candidates, housenum)

        return res

    get_val_udf = udf(get_val_2)
    matched_violations = violations_df.withColumn('PHYSICALID', 
                                                get_val_udf(violations_df['v.COUNTY'], violations_df['v.STREETNAME'], 
                                                                            violations_df['v.NUM0'], violations_df['v.NUM1']))
    matched_violations = matched_violations.filter( matched_violations['PHYSICALID'].isNotNull() )
    matched_violations = matched_violations.withColumn("PHYSICALID", matched_violations["PHYSICALID"].cast("integer"))
    matched_violations = matched_violations.orderBy("PHYSICALID")
    matched_violations = matched_violations.groupBy("PHYSICALID", "YEAR").agg(count("*").alias("YEAR_COUNT"))
    matched_violations.createOrReplaceTempView("matched_violations")

    summaries = spark.sql(
        "select PHYSICALID, " +
        "MAX(CASE WHEN (YEAR = 2015) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2015, " +
        "MAX(CASE WHEN (YEAR = 2016) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2016, " +
        "MAX(CASE WHEN (YEAR = 2017) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2017, " +
        "MAX(CASE WHEN (YEAR = 2018) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2018, " +
        "MAX(CASE WHEN (YEAR = 2019) THEN YEAR_COUNT ELSE 0 END) AS COUNT_2019  " +
        "from matched_violations " +
        "group by PHYSICALID " +
        "order by PHYSICALID "
    )
    
    getOLS_udf = udf(getOLS)
    summaries = summaries.withColumn('OLS_COEF', 
                    getOLS_udf(array('COUNT_2015', 'COUNT_2016', 'COUNT_2017', 'COUNT_2018', 'COUNT_2019')))

    streets_df = streets_df.select(col("s.PHYSICALID")) \
                        .join(summaries, "PHYSICALID", how='left') \
                        .distinct() \
                        .orderBy("PHYSICALID") \

    streets_df = streets_df.withColumn("COUNT_2015",coalesce("COUNT_2015", lit(0))) 
    streets_df = streets_df.withColumn("COUNT_2016",coalesce("COUNT_2016", lit(0))) 
    streets_df = streets_df.withColumn("COUNT_2017",coalesce("COUNT_2017", lit(0))) 
    streets_df = streets_df.withColumn("COUNT_2018",coalesce("COUNT_2018", lit(0))) 
    streets_df = streets_df.withColumn("COUNT_2019",coalesce("COUNT_2019", lit(0))) 
    streets_df = streets_df.withColumn("OLS_COEF",  coalesce("OLS_COEF", lit(0.0))) 

    streets_df.write.csv(output_file, header=False)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    
    parser.add_argument("output_path", type=Path)
    p = parser.parse_args()

    print("Output Path: ", str(p.output_path))
    starttime = datetime.now()
    run_spark(str(p.output_path))
    elapsed = datetime.now() - starttime
    print("Done, Elapsed: {} (secs)".format(elapsed.total_seconds()))
