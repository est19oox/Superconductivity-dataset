#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pyspark
import os
import sys
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
plt.style.use(style='seaborn')

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import types as T
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Regression').getOrCreate()
spark


# In[2]:


df = spark.read.csv('train.csv', header = True, inferSchema = True)
dfPySpark = df.select('mean_atomic_mass','mean_fie','mean_atomic_radius','mean_Density','mean_ElectronAffinity','mean_FusionHeat','mean_ThermalConductivity','mean_Valence','critical_temp')
dfPySpark.show()


# In[3]:


dfPySpark = df.select('mean_atomic_mass','mean_fie','mean_atomic_radius','mean_Density','mean_ElectronAffinity','mean_FusionHeat','mean_ThermalConductivity','mean_Valence','critical_temp')
dfPySpark.show()


# In[4]:


from pyspark.sql.functions import col, count, isnan, when
dfPySpark.select([count(when(col(c).isNull(), c)).alias(c) for c in dfPySpark.columns]).show()


# In[5]:


features = dfPySpark.drop('critical_temp')


# In[6]:


myAssembler = VectorAssembler(
    inputCols = features.columns,
    outputCol = 'features'
)

output = myAssembler.transform(dfPySpark).select('features','critical_temp')
output.show(truncate = False)


# In[7]:


(test, train) = output.randomSplit([0.33, 0.66], seed = 9)

# Print the dataset
print(f'Train data set size: {train.count()} records')
print(f'Test data set size: {test.count()} records')


# In[8]:


train.printSchema()
test.printSchema()


# #### Linear Regression

# In[9]:


from pyspark.ml.regression import LinearRegression
linReg = LinearRegression(featuresCol = 'features', labelCol = 'critical_temp')
linearModel = linReg.fit(train)


# In[10]:


print('Coefficients: ' + str(linearModel.coefficients))
print('\nIntercept: ' + str(linearModel.intercept))


# In[11]:


trainSummary = linearModel.summary
print('RMSE: %f' % trainSummary.rootMeanSquaredError)
print('\nr2: %f' % trainSummary.r2)


# In[12]:


from pyspark.sql.functions import abs
predictions = linearModel.transform(test)
x = ((predictions['critical_temp']-predictions['prediction'])/predictions['critical_temp']) * 100
predictions = predictions.withColumn('Accuracy', abs(x))
predictions.select('prediction', 'critical_temp', 'Accuracy', 'features').show(truncate = False)


# In[13]:


from pyspark.ml.evaluation import RegressionEvaluator
myPredEvaluator = RegressionEvaluator (
    predictionCol = 'prediction',
    labelCol = 'critical_temp',
    metricName = 'r2'
)

print('R Squared (R2) on test data = %g' % myPredEvaluator.evaluate(predictions))


# In[14]:


def adjR2(x):
    r2 = trainSummary.r2
    n = dfPySpark.count()
    p = len(dfPySpark.columns)
    adjustedR2 = 1 - (1 - r2) * (n - 1) / (n-p-1)
    return adjustedR2


# In[15]:


adjR2(train)


# In[16]:


adjR2(test)


# In[17]:


linReg = LinearRegression(
    featuresCol = 'features', 
    labelCol = 'critical_temp',
    maxIter = 50,
    regParam = 0.12,
    elasticNetParam = 0.2)
linearModel = linReg.fit(train)


# In[18]:


linearModel.summary.rootMeanSquaredError


# In[19]:


colName = features.columns
featuresRDD = features.rdd
featuresRDD.collect()


# In[20]:


featuresRDD = features.rdd.map(lambda row: row[0:])
featuresRDD.collect()


# In[21]:


from pyspark.mllib.feature import StandardScaler
scaler1 = StandardScaler().fit(featuresRDD)
scaledFeatures = scaler1.transform(featuresRDD)


# In[22]:


for data in scaledFeatures.collect():
    print(data)


# In[23]:


df2 = scaledFeatures.map(lambda x: (x, )).toDF(['ScaledData'])


# In[24]:


df2.show(truncate = False)


# #### Random Forest Regressor

# In[25]:


# Random Forest
from pyspark.ml.regression import RandomForestRegressor
randomForestReg = RandomForestRegressor(featuresCol = 'features', labelCol = 'critical_temp')
randomForestModel = randomForestReg.fit(train)


# In[26]:


predictions = randomForestModel.transform(test)
predictions.show()


# In[27]:


from pyspark.sql.functions import abs
predictions = randomForestModel.transform(test)
x = ((predictions['critical_temp']-predictions['prediction'])/predictions['critical_temp']) * 100
predictions = predictions.withColumn('Accuracy', abs(x))
predictions.select('prediction', 'critical_temp', 'Accuracy', 'features').show(truncate = False)


# In[28]:


from pyspark.ml.evaluation import RegressionEvaluator
myPredEvaluator = RegressionEvaluator (
    predictionCol = 'prediction',
    labelCol = 'critical_temp',
    metricName = 'rmse'
)

print('Root Mean Square Error (rmse) on test data = %g' % myPredEvaluator.evaluate(predictions))


# In[29]:


myPredEvaluator = RegressionEvaluator (
    predictionCol = 'prediction',
    labelCol = 'critical_temp',
    metricName = 'r2'
)

print('R Squared (R2) on test data = %g' % myPredEvaluator.evaluate(predictions))


# #### Logistic Regression

# In[30]:


# Import logistic Regression model
from pyspark.ml.classification import LogisticRegression


# In[31]:


output.show(200, truncate=False)


# In[32]:


#output.withColumn("critical_temp",col("critical_temp").cast("Integer")).show()
#output.withColumn("critical_temp",col("critical_temp")*0).show()

df_stats = output.select(
    F.mean(col('critical_temp')).alias('mean')
).collect()
mean = df_stats[0]['mean']
print(mean)

output.withColumn("critical_temp", F.when(F.col("critical_temp")>mean, 1).otherwise(0)).show()


# In[33]:


data = output.withColumn("critical_temp", F.when(F.col("critical_temp")>mean, 1).otherwise(0)).select (
    F.col('features').alias('features'),
    F.col('critical_temp').alias('label')
)


# In[34]:


model = LogisticRegression().fit(data)


# In[35]:


model.summary.areaUnderROC


# In[36]:


model.summary.pr.show()


# In[ ]:





# In[ ]:




