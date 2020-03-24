# Databricks notebook source
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder.config(conf = SparkConf()).getOrCreate()

from pyspark.ml.linalg import Vector,Vectors
from pyspark.sql.types import DoubleType, StructType, StructField,StringType
from pyspark.sql import Row,functions
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel,BinaryLogisticRegressionSummary,LogisticRegression
import mlflow

# COMMAND ----------

##Connection with ADLS to get training Data
##Will move to cluster.init script

spark.conf.set("fs.azure.account.key.databrickspocadlsgen201.dfs.core.windows.net", "hLcaEZFjjIFml3RE5IQhPaeDznU++tDcZnjqx5QeyR0EkOeCUFNZ7FRIoFDYbGSFd9lRtqS52Dg1VM4j7I5Cvw==")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "true")
dbutils.fs.ls("abfss://sparktest@databrickspocadlsgen201.dfs.core.windows.net/")
spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization", "false")

dbutils.fs.ls("abfss://trainingset@databrickspocadlsgen201.dfs.core.windows.net/")

# COMMAND ----------

#mount modelbase blob storage

dbutils.fs.mount(
source = "wasbs://modelbase@databrockspocblob01.blob.core.windows.net",
mount_point = "/mnt/blobmount01",
extra_configs = {"fs.azure.sas.modelbase.databrockspocblob01.blob.core.windows.net":"?st=2020-03-18T09%3A53%3A36Z&se=2020-04-19T09%3A53%3A00Z&sp=rwdl&sv=2018-03-28&sr=c&sig=IGruB%2FtXup1RjdhHeIIUX3Zs2zmKdCORKeON%2FI%2FKN1o%3D"})


#%sh
#ls -l /dbfs/mnt/blobmount01
#dbutils.fs.unmount("/mnt/blobmount01")

# COMMAND ----------

with mlflow.start_run(experiment_id='746239282700942',run_name='IrisLearning0323_v1') as run:
  # Log a parameter (key-value pair)
  mlflow.log_param("param1", 5)
  # Log a metric; metrics can be updated throughout the run
  mlflow.log_metric("foo", 2, step=1)
  mlflow.log_metric("foo", 4, step=2)
  mlflow.log_metric("foo", 6, step=3)

# COMMAND ----------

schema = StructType([
    StructField("_c0", DoubleType(), True),
    StructField("_c1", DoubleType(), True),
    StructField("_c2", DoubleType(), True),
    StructField("_c3", DoubleType(), True),
    StructField("_c4", StringType(), True)])

data = spark.read.csv("abfss://trainingset@databrickspocadlsgen201.dfs.core.windows.net/iris/iris.txt",schema=schema)
data.show(5)

# COMMAND ----------

df_assembler = VectorAssembler(inputCols=['_c0','_c1','_c2',\
                                         '_c3'], outputCol='features')
data = df_assembler.transform(data).select('features','_c4')
data.show(10,truncate=False)

# COMMAND ----------

labelIndexer = StringIndexer().setInputCol("_c4"). \
    setOutputCol("indexedLabel").fit(data)
data = labelIndexer.transform(data)
data.show(5)

# COMMAND ----------

featureIndexer = VectorIndexer(maxCategories=5).setInputCol("features"). \
    setOutputCol("indexedFeatures").fit(data)
data = featureIndexer.transform(data)
data.show(5)

# COMMAND ----------

trainData, testData = data.randomSplit([0.7, 0.3])
trainData.show(5)

# COMMAND ----------

lr = LogisticRegression(labelCol='indexedLabel',featuresCol='indexedFeatures',\
                        maxIter=100, regParam=0.3, elasticNetParam=0.8).fit(trainData)
testData = lr.transform(testData)
testData .show(5)

# COMMAND ----------

labelConverter = IndexToString(inputCol='prediction',outputCol='predictedLabel',\
                              labels=labelIndexer.labels)
testData = labelConverter.transform(testData)
testData.show(5)

# COMMAND ----------

lr.save("abfss://trainingset@databrickspocadlsgen201.dfs.core.windows.net/iris/lr_")

# COMMAND ----------

### 读取鸢尾花数据集
schema = StructType([
    StructField("_c0", DoubleType(), True),
    StructField("_c1", DoubleType(), True),
    StructField("_c2", DoubleType(), True),
    StructField("_c3", DoubleType(), True),
    StructField("_c4", StringType(), True)])
data = spark.read.csv("abfss://trainingset@databrickspocadlsgen201.dfs.core.windows.net/iris/iris.txt",schema=schema)
# data.show(5)

labelIndexer = StringIndexer().setInputCol("_c4"). \
    setOutputCol("indexedLabel").fit(data)
data = labelIndexer.transform(data)
# data.show()

trainData, testData = data.randomSplit([0.7, 0.3])

assembler = VectorAssembler(inputCols=['_c0','_c1','_c2','_c3'], outputCol='features')

featureIndexer = VectorIndexer().setInputCol('features'). \
    setOutputCol("indexedFeatures")


lr = LogisticRegression().\
    setLabelCol("indexedLabel"). \
    setFeaturesCol("indexedFeatures"). \
    setMaxIter(100). \
    setRegParam(0.3). \
    setElasticNetParam(0.8)
# print("LogisticRegression parameters:\n" + lr.explainParams())

labelConverter = IndexToString(). \
    setInputCol("prediction"). \
    setOutputCol("predictedLabel"). \
    setLabels(labelIndexer.labels)
lrPipeline = Pipeline(). \
    setStages([assembler, featureIndexer, lr, labelConverter])

lrPipelineModel = lrPipeline.fit(trainData)

# COMMAND ----------

mlflow.end_run(status='FINISHED')

# COMMAND ----------

## 保存模型
lrPipelineModel.save("/mnt/blobmount01/iris20200318")