import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import rtree
import argparse
from pathlib import Path
from pyspark import SparkContext
from geopandas import GeoDataFrame


def createIndex(shapefile):
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return {"index": index, "zones": zones}

def findZone(p, geo_map):
    match = geo_map['index'].intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if geo_map['zones'].geometry[idx].contains(p):
            return idx
    return None

def processTrips(pid, records):
    import csv
    import pyproj
    import shapely.geometry as geom

    # Skip the header
    if pid==0:
        next(records)
    
    reader = csv.reader(records)
    counts = {}
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    boros = createIndex('boroughs.geojson')    
    neighborhoods = createIndex('neighborhoods.geojson')    
    
    for row in reader:
        # 'tpep_pickup_datetime,tpep_dropoff_datetime,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude',
        try: 
            if 'NULL' in row[2:5]: 
                continue
            pickup_point = geom.Point(proj(float(row[3]), float(row[2])))
            start_boro = findZone(pickup_point, boros)
            
            if start_boro:
                boro_name = boros['zones'].iloc[start_boro]['boroname']

                dropoff_point= geom.Point(proj(float(row[5]), float(row[4])))
                end_hood = findZone(dropoff_point, neighborhoods)
                if end_hood:
                    hood_name = neighborhoods['zones'].iloc[end_hood]['neighborhood']
                    yield ( (boro_name, hood_name), 1 )
        except: 
            print("Failed at: ", row) ##TODO this won't log anything

def run_spark(taxi_file):
    sc = SparkContext()

    from heapq import nlargest
    from operator import itemgetter

    start = time.time()
    rdd = sc.textFile(taxi_file).mapPartitionsWithIndex(processTrips).cache()
    
    counts = rdd.reduceByKey(lambda x,y: x+y) \
                .map(lambda x: ( x[0][0], [(x[0][1], x[1])] ) ) \
                .reduceByKey(lambda x,y: x+y) \
                .mapValues(lambda hood_counts: nlargest(3, hood_counts, key=itemgetter(1))) \
                .map(lambda x: x[0]+ "," + ",".join([str(i) for sub in x[1] for i in sub])) \
                .collect()
    
    return counts.sort()
    print("Execution Time(secs): ", time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)

    p = parser.parse_args()
    results = run_spark(str(p.input_file))
    results
