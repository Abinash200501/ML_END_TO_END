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
curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
sudo apt-get install unzip
unzip awscli-bundle.zip
sudo apt-get install python3-venv
mkdir mlflow
cd mlflow
python -m venv myenv
source myenv/bin/activate
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

## Setup For GitHub Actions

* Create an IAM user with following access:
	- AmazonEC2ContainerRegistryFullAccess
	- AmazonEC2FullAccess
* Create an ECR repository and copy the URI
* Create an EC2 instance with Ubuntu t2.micro. After creating the machine run the following Commands:
	```bash
	sudo apt-get update -y
	sudo apt-get upgrade -y
	curl -fsSL https://get.docker.com -o get-docker.sh
	sudo sh get-docker.sh
	sudo usermod -aG docker ubuntu
	newgrp docker
	sudo apt-get install python3-pip
	pip install dvc==3.45.0 dvc[s3] boto3==1.34.34
	pip install pyOpenSSL --upgrade
	```

* Install AWS CLI:
	```bash
	curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
	sudo apt-get install unzip
	unzip awscli-bundle.zip
	sudo apt-get install python3-venv
	sudo ./awscli-bundle/install -i /usr/local/aws -b /usr/local/bin/aws
	aws Configure #input the IAM user credentials
	```
* Restart the EC2 machine
* Finally, in the Security Group of the EC2 Machine add a new rule to access the docker server 0.0.0.0/0 on port 8000

* Go to (in GitHub Repo) settings>actions>runner>new self hosted runner>linux
* A list of commands will be shown on the github page copy and paste it in the EC2 machine one by one
* Setup The Following secrets:
	```bash
	AWS_ACCESS_KEY_ID=
	AWS_SECRET_ACCESS_KEY=
	AWS_REGION= 
	ECR_REPOSITORY_UI= 
	ECR_REPOSITORY_NAME= 
	MLFLOW_TRACKING_URI=