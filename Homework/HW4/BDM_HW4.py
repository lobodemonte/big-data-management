import pandas as pd
import numpy as np
import rtree
from pyspark import SparkContext
import traceback

boros_file = 'boroughs.geojson'
neighborhood_file = 'neighborhoods.geojson'

def createIndex(shapefile):
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)

def findZone(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None

def processTrips(pid, records):
    import csv
    import pyproj
    import shapely.geometry as geom
    
    reader = csv.reader(records)
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    
    index_n, neighborhoods = createIndex(neighborhood_file)  
    
    # Skip the header
    if pid == 0:
        next(records)

    counts = {}
    for row in reader:
        try: 
            # if 'NULL' in row[2:6]: 
            #     continue
            if 'NULL' in row[5:7] or 'NULL' in row[9:11]:
                continue

            #pickup_point = geom.Point(proj(float(row[3]), float(row[2])))
            #dropoff_point= geom.Point(proj(float(row[5]), float(row[4])))
            pickup_point = geom.Point(proj(float(row[5]), float(row[6])))
            dropoff_point= geom.Point(proj(float(row[9]), float(row[10])))

            start_idx = findZone(pickup_point, index_n, neighborhoods) 
            end_idx   = findZone(dropoff_point, index_n, neighborhoods)
            
            if start_idx and end_idx:
                borough = neighborhoods.iloc[start_idx]['borough']
                neighborhood = neighborhoods.iloc[end_idx]['neighborhood']
                counts[(borough,neighborhood)] = counts.get((borough,neighborhood), 0) + 1

        except: 
            print("Failed at: ", row) ## TODO this won't log anything
            print(traceback.format_exc())

    return counts.items()

def run_spark(taxi_file, output_path):
    
    sc = SparkContext()
    rdd = sc.textFile(taxi_file)

    counts_rdd = rdd.mapPartitionsWithIndex(processTrips) \
                .reduceByKey(lambda x, y: x + y ) \
                .map(lambda x: ( x[0][0], [(x[0][1], x[1])] ) ) \
                .reduceByKey(lambda x, y: x + y ) \
                .mapValues(lambda hood_counts: sorted(hood_counts, reverse=True, key=lambda tup:tup[1])[:3]) \
                .sortByKey() \
                .map(lambda x: str(x[0]) + "," + ",".join([str(i) for sub in x[1] for i in sub])) \
   
    counts_rdd.saveAsTextFile(output_path)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_path", type=Path)
    p = parser.parse_args()

    print("Input File: ", str(p.input_file))
    print("Output Path: ", str(p.output_path))
    run_spark(str(p.input_file), str(p.output_path))
    print("Done")
