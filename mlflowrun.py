import dagshub
dagshub.init(repo_owner='jadhavgaurav',
             repo_name='Kidney_disease_classification_cnn',
             mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)