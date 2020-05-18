import pandas as pd
import numpy as np
import time
from pyspark import SparkContext
import traceback
from datetime import datetime

# tweets_file = "tweets-sample.csv"
# cities_file = "500cities_tracts.geojson"
# sched_drugs_file = "drug_sched2.txt"
# illegal_drugs_file = "drug_illegal.txt"

tweets_file = "hdfs:///tmp/bdm/tweets-100m.csv"
cities_file = "500cities_tracts.geojson"
illegal_drugs_file = "drug_illegal.txt"
sched_drugs_file = "drug_sched2.txt"

ZONES_B = None

def createIndex(shapefile):
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(5070))
    #zones = ZONES_B.value
    index = rtree.Rtree()
    for idx, geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)

def findZone(p, index, zones):
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None

# 450845003896479744|
# 33.01608281|-97.30442766|
# hungrypoop|
# Tue Apr 01 03:59:59 +0000 2014|
# i wish i could play the moog synthesizer|
# could moog play synthesizer the wish


def processTweets(pid, raw_tweets):
    import pyproj
    import shapely.geometry as geom
    drug_words = set(line.strip() for line in open(sched_drugs_file))
    drug_words.update(set(line.strip() for line in open(illegal_drugs_file)))
    
    proj = pyproj.Proj(init="epsg:5070", preserve_units=True)
    index, zones = createIndex("500cities_tracts.geojson")

    results = {}
    for raw_tweet in raw_tweets:
        elems = raw_tweet.strip().split("|")
        words = set(elems[-1].lower().split(" "))
        drug_related = drug_words.intersection(words)
        if len(drug_related) > 0:
            try: 
                point = geom.Point(proj(float(elems[2]), float(elems[1])))
                idx = findZone(point, index, zones)
                if idx:
                    tract_id = zones.plctract10[idx]
                    pop = zones.plctrpop10[idx]
                    if pop > 0:
                        results[tract_id] = results.get(tract_id, 0.0) + 1.0 / pop
            except:
                ## I just need an output tbh
                continue
    
    return results.items()

def run_spark(sc, output_path):
    rdd = sc.textFile(tweets_file)

    results_rdd = rdd.mapPartitionsWithIndex(processTweets) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda x: x[0])
    
    results_rdd.saveAsTextFile(output_path)

    
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import geopandas as gpd
    import fiona.crs

    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    p = parser.parse_args()
    print("Output Path: ", str(p.output_path))
    starttime = datetime.now()
    print("Start Time: ", starttime)

    sc = SparkContext()

    #zones = gpd.read_file(cities_file).to_crs(fiona.crs.from_epsg(5070))
    #ZONES_B = sc.broadcast(zones)
    run_spark(sc, str(p.output_path))
    elapsed = datetime.now() - starttime
    print("Done, Elapsed: {} (secs)".format(elapsed.total_seconds()))
