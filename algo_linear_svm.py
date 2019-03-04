from pyspark.ml.classification import LinearSVC

# Load training data
from pyspark.shell import spark

training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvc_model = lsvc.fit(training)

# Print the coefficients and intercept for linear SVC
print("Coefficients: " + str(lsvc_model.coefficients))
print("Intercept: " + str(lsvc_model.intercept))
