import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import udf
from transformers import pipeline

# Limit thread usage (important for HuggingFace on CPU)
os.environ["OMP_NUM_THREADS"] = "10" # Should be adjusted based on CPU cores
os.environ["PYSPARK_PYTHON"] = "/home/amar/Desktop/2nd_sem/MLOps/bin/python" # Path to your Python environment

# Step 1: Parse .txt file
def parse_reviews(file_path):
    reviews = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_reviews = content.strip().split("\n\n")
    for block in raw_reviews:
        review_score = None
        review_text = None
        for line in block.split("\n"):
            if line.startswith("review/score:"):
                review_score = line.split("review/score:")[1].strip()
            elif line.startswith("review/text:"):
                review_text = line.split("review/text:")[1].strip()
        if review_score and review_text:
            reviews.append((review_score, review_text))
    return reviews

parsed_data = parse_reviews("Gourmet_Foods.txt")

# Step 2: Create Spark Session with limited parallelism
spark = SparkSession.builder \
    .appName("SentimentAnalysisGourmet") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.default.parallelism", "10") \
    .getOrCreate()

# Reduce Spark log level
spark.sparkContext.setLogLevel("ERROR")

# Schema
schema = StructType([
    StructField("star_rating", StringType(), True),
    StructField("review_body", StringType(), True)
])

# TEST with a sample
df = spark.createDataFrame(parsed_data, schema=schema)  # Increase to full data once stable

# Step 3: Sentiment classifier
sentiment_model = pipeline("sentiment-analysis")
broadcast_model = spark.sparkContext.broadcast(sentiment_model)

def classify_sentiment(text):
    try:
        result = broadcast_model.value(text[:512])
        return result[0]['label']
    except:
        return "ERROR"

sentiment_udf = udf(classify_sentiment, StringType())

# Step 4: Apply UDF
df_with_sentiment = df.withColumn("predicted_sentiment", sentiment_udf(df["review_body"]))

# Step 5: Save result
df_with_sentiment.write.csv("sentiment_output", header=True, mode="overwrite")

spark.stop()
