import geopandas as gpd
import pandas as pd
import numpy as np
import rtree
from pyspark import SparkContext

boros_file = 'boroughs.geojson'
neighborhood_file = 'neighborhoods.geojson'

def genIndex(shapefile):
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx, geometry in enumerate(zones.geometry):
        yield (idx, geometry.bounds, zones.iloc[idx])

def getZone(p, index, field):
    matches = index.intersection((p.x, p.y, p.x, p.y), objects=True)
    for match in matches:
        hood = match.object
        if hood.geometry.contains(p):
            return hood[field]
    return None

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
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    
    # boros = createIndex(boros_file)    
    # neighborhoods = createIndex(neighborhood_file)    
    
    boros_index = rtree.index.Index(genIndex('boroughs.geojson'))
    hood_index = rtree.index.Index(genIndex('neighborhoods.geojson'))

    for row in reader:
        # 'tpep_pickup_datetime,tpep_dropoff_datetime,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude',
        try: 
            if 'NULL' in row[2:5]: 
                continue
            # if 'NULL' in row[5:6] or 'NULL' in row[9:10]:
            #     continue

            pickup_point = geom.Point(proj(float(row[3]), float(row[2])))
            # pickup_point = geom.Point(proj(float(row[5]), float(row[6])))            

#             start_boro = findZone(pickup_point, boros)
            start_boro = getZone(pickup_point, boros_index, 'boroname')
            
            if start_boro:
                dropoff_point= geom.Point(proj(float(row[5]), float(row[4])))
                # dropoff_point= geom.Point(proj(float(row[9]), float(row[10])))
                
                # end_hood = findZone(dropoff_point, neighborhoods)
                end_hood = getZone(dropoff_point, hood_index, 'neighborhood')
                
                if end_hood:
                    # boro_name = boros['zones'].iloc[start_boro]['boroname']
                    # hood_name = neighborhoods['zones'].iloc[end_hood]['neighborhood']
                    # yield ( (boro_name, hood_name), 1 )
                    yield ( (start_boro, end_hood), 1 )

        except: 
            print("Failed at: ", row) ##TODO this won't log anything

def run_spark(taxi_file, sc):
    from heapq import nlargest
    from operator import itemgetter

    rdd = sc.textFile(taxi_file).mapPartitionsWithIndex(processTrips).cache()
    
    counts = rdd.reduceByKey(lambda x,y: x+y) \
                .map(lambda x: ( x[0][0], [(x[0][1], x[1])] ) ) \
                .reduceByKey(lambda x,y: x+y) \
                .mapValues(lambda hood_counts: nlargest(3, hood_counts, key=itemgetter(1))) \
                .map(lambda x: x[0]+ "," + ",".join([str(i) for sub in x[1] for i in sub])) \
                .collect()
    
    counts.sort()
    return counts

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    sc = SparkContext()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    p = parser.parse_args()

    print("Input File: ", str(p.input_file))
    results = run_spark(str(p.input_file), sc)
    
    print("Results")
    print(results)

    import index from rtree
    idx = index.Index()

