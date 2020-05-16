import pandas as pd
import numpy as np
import traceback
import time

from datetime import datetime
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array, count, broadcast, coalesce, lit

def to_upper(string):
    if string is not None:
        return string.strip().upper()
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
        if county == 'R' or county == 'ST':
            return 5
    return None

def get_year(date_string):
    if date_string is not None:
        data_val = datetime.strptime(date_string.strip(), '%m/%d/%Y')    
        return data_val.year
    return None

def get_street_number(street_val):
    if street_val is not None:
        if type(street_val) is int:
            return street_val
        elems = street_val.split("-")  
        if len(elems) == 1 and elems[0].isdigit():
            return int(elems[0])
        elif len(elems) == 2 and elems[0].isdigit() and elems[1].isdigit():
            new_val = elems[0] + "{:04d}".format(int(elems[1]))
            if new_val.isdigit():
                return int(new_val)          
        else:
            new_val = "".join(elems)
            if new_val.isdigit():
                return int(new_val)
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

    violations_df = violations_df.withColumnRenamed("Violation County","COUNTY")
    violations_df = violations_df.withColumnRenamed("House Number","HOUSENUM")
    violations_df = violations_df.withColumnRenamed("Street Name","STREETNAME")
    violations_df = violations_df.withColumnRenamed("Issue Date","YEAR")
    
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

    streets_df = streets_df.withColumnRenamed("L_LOW_HN","OddLo")
    streets_df = streets_df.withColumnRenamed("L_HIGH_HN","OddHi")
    streets_df = streets_df.withColumnRenamed("R_LOW_HN","EvenLo")
    streets_df = streets_df.withColumnRenamed("R_HIGH_HN","EvenHi")
    
    streets_df = streets_df.filter((streets_df['BOROCODE'].isNotNull()) & (streets_df['PHYSICALID'].isNotNull()))
    
    streets_df = streets_df.withColumn('BOROCODE', as_digit_udf(streets_df['BOROCODE']))
    streets_df = streets_df.withColumn('FULL_STREE', to_upper_udf(streets_df['FULL_STREE']))
    streets_df = streets_df.withColumn('ST_LABEL',   to_upper_udf(streets_df['ST_LABEL']))
    streets_df = streets_df.withColumn('OddLo', get_street_number_udf(streets_df['OddLo']))
    streets_df = streets_df.withColumn('OddHi', get_street_number_udf(streets_df['OddHi']))
    streets_df = streets_df.withColumn('EvenLo', get_street_number_udf(streets_df['EvenLo']))
    streets_df = streets_df.withColumn('EvenHi', get_street_number_udf(streets_df['EvenHi']))
    
    streets_df = streets_df.withColumn("BOROCODE", streets_df["BOROCODE"].cast("integer"))
    streets_df = streets_df.withColumn("OddLo", streets_df["OddLo"].cast("integer"))
    streets_df = streets_df.withColumn("OddHi", streets_df["OddHi"].cast("integer"))
    streets_df = streets_df.withColumn("EvenLo", streets_df["EvenLo"].cast("integer"))
    streets_df = streets_df.withColumn("EvenHi", streets_df["EvenHi"].cast("integer"))
    
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

def run_spark(output_path):
    sc = SparkContext()
    spark = SparkSession(sc)
    
    # streets = "nyc_cscl.csv"
    # violations = "nyc_parking_violation/*.csv"
    streets = "hdfs:///tmp/bdm/nyc_cscl.csv"
    violations = "hdfs:///tmp/bdm/nyc_parking_violation/*.csv"

    violations_df = get_violations_df(violations, spark)
    streets_df = get_streets_df(streets, spark)

    streets_dict = streets_df.rdd.flatMap(mapper).reduceByKey(lambda x,y: x+y).collectAsMap()
    streets_dict_bc = sc.broadcast(streets_dict)

    def get_val(borocode, street, housenum):
        val = streets_dict_bc.value.get( (borocode, street) )
        if val:
            for item in val:
                if housenum % 2 == 0:
                    if item[0] >= housenum and housenum <= item[1]:
                        return item[4]
                else:
                    if item[2] >= housenum and housenum <= item[3]:
                        return item[4]     
        return None

    get_val_udf = udf(get_val)
    matched_violations = violations_df.withColumn('PHYSICALID', get_val_udf(violations_df['v.COUNTY'], 
                                                                            violations_df['v.STREETNAME'], 
                                                                            violations_df['v.HOUSENUM']
                                                            ))

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
    
    streets_df.write.csv(output_path, header=False)

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
