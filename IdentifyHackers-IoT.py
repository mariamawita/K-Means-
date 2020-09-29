#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:52:29 2020

@author: mariamawitanteneh
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.appName('k-means clustering').getOrCreate()
# Loads data
dataset = spark.read.csv("/Users/mariamawitanteneh/Desktop/IOT/hack_data.csv",header=True,inferSchema=True)
dataset.head()

"""
Out[25]: Row(Session_Connection_Time=8.0, Bytes Transferred=391.09, Kali_Trace_Used=1, Servers_Corrupted=2.96, 
Pages_Corrupted=7.0, Location='Slovenia', WPM_Typing_Speed=72.37)
"""

dataset.describe().show()

"""
+-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
|summary|Session_Connection_Time| Bytes Transferred|   Kali_Trace_Used|Servers_Corrupted|   Pages_Corrupted|   Location|  WPM_Typing_Speed|
+-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
|  count|                    334|               334|               334|              334|               334|        334|               334|
|   mean|     30.008982035928145| 607.2452694610777|0.5119760479041916|5.258502994011977|10.838323353293413|       null|57.342395209580864|
| stddev|     14.088200614636158|286.33593163576757|0.5006065264451406| 2.30190693339697|  3.06352633036022|       null| 13.41106336843464|
|    min|                    1.0|              10.0|                 0|              1.0|               6.0|Afghanistan|              40.0|
|    max|                   60.0|            1330.5|                 1|             10.0|              15.0|   Zimbabwe|              75.0|
+-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
"""

# ## Format the Data
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

print(dataset.columns)

"""
['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 
'Pages_Corrupted', 'Location', 'WPM_Typing_Speed']
"""
# Without Location
feature_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used', 'Servers_Corrupted', 
                'Pages_Corrupted', 'WPM_Typing_Speed']

vec_assembler = VectorAssembler(inputCols = feature_cols, outputCol='features')
feature_data = vec_assembler.transform(dataset)

from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
# Computing summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(feature_data)
# Normalizing each feature to have unit standard deviation.
final_data = scalerModel.transform(feature_data)
final_data
"""
Out[31]: DataFrame[Session_Connection_Time: double, Bytes Transferred: double, Kali_Trace_Used: int, Servers_Corrupted: double, 
Pages_Corrupted: double, Location: string, WPM_Typing_Speed: double, features: vector, scaledFeatures: vector]
"""

from pyspark.ml.clustering import KMeans
#training the model and evaluaing with k=2 and k=3 to see which would be the correct one 
kmeans2 = KMeans(featuresCol='scaledFeatures', k=2)
model2 = kmeans2.fit(final_data)

kmeans3 = KMeans(featuresCol='scaledFeatures', k=3)
model3 = kmeans3.fit(final_data)


wssse_k2 = model2.computeCost(final_data)
print("When k=2")
print("Within Set Sum of Squared Errors = " + str(wssse_k2))
"""
When k=2
Within Set Sum of Squared Errors = 601.7707512676716
"""

wssse_k3 = model3.computeCost(final_data)
print("When k=2")
print("Within Set Sum of Squared Errors = " + str(wssse_k3))
"""
When k=2
Within Set Sum of Squared Errors = 434.1492898715845
"""

"""
we do no get much to conclude whether it is 3 or 2 attackers from thee wssse. This is because we
know tha  as k increases hee wssse decreases as in this case. If he analysis is coninues by 
increasing k, wssse will continue to decrease. 
"""
centers2 = model2.clusterCenters()
print("Cluster Centers: ")
for center in centers2:
    print(center)
    
"""
Cluster Centers: 
[1.26023837 1.31829808 0.99280765 1.36491885 2.5625043  5.26676612]
[2.99991988 2.92319035 1.05261534 3.20390443 4.51321315 3.28474   ]
"""
    
centers3 = model3.clusterCenters()
print("Cluster Centers: ")
for center in centers3:
    print(center)
    
"""
Cluster Centers: 
[1.21780112 1.37901802 1.99757683 1.37198977 2.55237797 5.29152222]
[2.99991988 2.92319035 1.05261534 3.20390443 4.51321315 3.28474   ]
[1.30217042 1.25830099 0.         1.35793211 2.57251009 5.24230473]
"""

# Predicting the label of each seed
model2.transform(final_data).select('prediction').show(5)
""" 
+----------+
|prediction|
+----------+
|         0|
|         0|
|         0|
|         0|
|         0|
+----------+
only showing top 5 rows
"""

model3.transform(final_data).select('prediction').show(5)
"""
+----------+
|prediction|
+----------+
|         0|
|         2|
|         0|
|         0|
|         2|
+----------+
only showing top 5 rows
"""
clusters = model2.transform(final_data).select('*')
clusters.groupBy("prediction").count().orderBy(F.desc("count")).show()
clusters.show(5)
clusters_pd = clusters.toPandas()
"""
+----------+-----+
|prediction|count|
+----------+-----+
|         0|  167|
|         1|  167|
+----------+-----+

+-----------------------+-----------------+---------------+-----------------+---------------+--------------------+----------------+--------------------+--------------------+----------+
|Session_Connection_Time|Bytes Transferred|Kali_Trace_Used|Servers_Corrupted|Pages_Corrupted|            Location|WPM_Typing_Speed|            features|      scaledFeatures|prediction|
+-----------------------+-----------------+---------------+-----------------+---------------+--------------------+----------------+--------------------+--------------------+----------+
|                    8.0|           391.09|              1|             2.96|            7.0|            Slovenia|           72.37|[8.0,391.09,1.0,2...|[0.56785108466505...|         0|
|                   20.0|           720.99|              0|             3.04|            9.0|British Virgin Is...|           69.08|[20.0,720.99,0.0,...|[1.41962771166263...|         0|
|                   31.0|           356.32|              1|             3.71|            8.0|             Tokelau|           70.58|[31.0,356.32,1.0,...|[2.20042295307707...|         0|
|                    2.0|           228.08|              1|             2.48|            8.0|             Bolivia|            70.8|[2.0,228.08,1.0,2...|[0.14196277116626...|         0|
|                   20.0|            408.5|              0|             3.57|            8.0|                Iraq|           71.28|[20.0,408.5,0.0,3...|[1.41962771166263...|         0|
+-----------------------+-----------------+---------------+-----------------+---------------+--------------------+----------------+--------------------+--------------------+----------+
only showing top 5 rows
"""

clusters = model3.transform(final_data).select('*')
clusters.groupBy("prediction").count().orderBy(F.desc("count")).show()
clusters.show(5)
clusters_pd = clusters.toPandas()
"""
+----------+-----+
|prediction|count|
+----------+-----+
|         1|  167|
|         2|   84|
|         0|   83|
+----------+-----+

+-----------------------+-----------------+---------------+-----------------+---------------+--------------------+----------------+--------------------+--------------------+----------+
|Session_Connection_Time|Bytes Transferred|Kali_Trace_Used|Servers_Corrupted|Pages_Corrupted|            Location|WPM_Typing_Speed|            features|      scaledFeatures|prediction|
+-----------------------+-----------------+---------------+-----------------+---------------+--------------------+----------------+--------------------+--------------------+----------+
|                    8.0|           391.09|              1|             2.96|            7.0|            Slovenia|           72.37|[8.0,391.09,1.0,2...|[0.56785108466505...|         0|
|                   20.0|           720.99|              0|             3.04|            9.0|British Virgin Is...|           69.08|[20.0,720.99,0.0,...|[1.41962771166263...|         2|
|                   31.0|           356.32|              1|             3.71|            8.0|             Tokelau|           70.58|[31.0,356.32,1.0,...|[2.20042295307707...|         0|
|                    2.0|           228.08|              1|             2.48|            8.0|             Bolivia|            70.8|[2.0,228.08,1.0,2...|[0.14196277116626...|         0|
|                   20.0|            408.5|              0|             3.57|            8.0|                Iraq|           71.28|[20.0,408.5,0.0,3...|[1.41962771166263...|         2|
+-----------------------+-----------------+---------------+-----------------+---------------+--------------------+----------------+--------------------+--------------------+----------+
only showing top 5 rows
"""

"""
K = 2 (attackers)
By using the transform and prediction we can check the conts of the attackes and if they are evenly
disributed or roughly the same amount of attack. This is an important facor because the engineer has
stated thatt the attackers trade off and would have similar amount of attact. From he above result 
we can see the prediction and the count. When k=2 the 2 attackers have an eevenly distributed count 
both 167, whereas when k=3 hee disribution seems far of with one being 167 and the others being 84
and 83. Therefore, it was 2 attackers (k=2), the algorithm actually shows equal distribution. This is true
because the engineer specifically said one of the key factors is just that, closely distributed 
attacks. the other part shhows the cluser for each column. 
"""
