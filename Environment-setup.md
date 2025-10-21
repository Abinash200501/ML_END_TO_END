# Environment setup for Github actions and self-hosted runner (AWS)

Create an IAM user with following permission:
1. EC2ContainerRegistryFullAccess
2. EC2FullAccess
3. S3FullAccess


## 1) S3-Bucket For mlflow and model tracking using DVC

* Create a S3 bucket with two folders one for mlflow tracking and model tracking using DVC

* For MlFlow Create an EC2 instance with Ubuntu t3.micro. After creating the machine run the following Commands

```bash
sudo apt update
sudo apt install python3-pip
sudo pip3 install pipenv
sudo pip3 install virtualenv
mkdir mlflow
cd mlflow
pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell
```
* Then set AWS credentials
```bash
aws configure #setup the IAM user accesskey, secret access key and region
```
* Finally, in the Security Group of the EC2 Machine add a new rule to access the mlflow server 0.0.0.0/0 on port 5000

```bash 
mlflow server -h 0.0.0.0 --default-artifact-root s3://dvc-remote-storage01 --allowed-hosts '*'
```

* For DVC to remote setup:
	* Install git, dvc, dvc[s3], boto3 , awscli locally
	```bash
	aws configure #setup IAM user credentials
	dvc remote add -d <remote-name> s3://dvc-remote-storage01/model
	dvc add saved_model/
	dvc push
	```