# Environment setup and to deploy on AWS server

This guide walks you through setting up the environment and deploying your ML system on AWS using **S3**, **EC2**, **ECR**, **MLflow**, **DVC**, and **GitHub Actions**.

## 1. Create IAM User and Permissions

Create an **IAM** user with the following access and with access key:
1. `AdministratorAccess`
2. `AmazonEC2ContainerRegistryFullAccess`
3. `AmazonEC2FullAccess`
4. `AmazonS3FullAccess`

----

# 2. S3 bucket for Mlflow and model tracking using DVC

* Create S3 bucket with two folders
    1. `mlflow/`
    2. `saved_model/` (for DVC-tracked models)

* Create an EC2 instance for mlflow using t3.micro and run the below commands after setup.

    * EC2 Setup:
        1. Choose Ubuntu as Amazon Machine Image (AMI) or anything based on preferences
        2. Allow inbound rules for **HTTP (80)**, **HTTPS (443)**, and **Custom TCP (5000)**.
        3. Attach an IAM role or configure credentials manually.
        4. Allocate sufficient storage (8–16 GB recommended).

    * Commands:

        ```bash
        sudo apt update
        sudo apt install python3-pip
        curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
        sudo apt-get install unzip
        unzip awscli-bundle.zip

        # Setup environment
        sudo apt-get install python3-venv
        mkdir mlflow
        cd mlflow
        python3 -m venv myenv
        source myenv/bin/activate

        # Install dependencies
        pip install mlflow
        pip install awscli
        pip install boto3
        ```

* Set AWS on that mlflow EC2 instance
    ```bash
    aws configure #setup the IAM user accesskey, secret access key and region
    ```
* Finally, in the Security Group of the EC2 Machine add a new rule to access the mlflow server 0.0.0.0/0 on port 5000

    ```bash 
    mlflow server --host 0.0.0.0 --default-artifact-root s3://mlops-bucket-storage-01/mlflow --allowed-hosts '*'
    ```
* DVC to Remote setup
    Install git, DVC, AWS-CLI, dvc[s3], boto3 locally
    ```bash
    aws configure #setup IAM user credentials
    dvc remote add -d <remote-name> s3://<s3 bucket-name>/saved_model
    dvc add saved_model/
    dvc push

# 3. Setup for GitHub actions

* Create an ECR Repository and copy the URI
* Create an EC2 instance with t3.micro or preffered instance type and run the following commands
    * Install docker
        ```bash
        sudo apt update
        sudo apt-get upgrade -y
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker ubuntu
        newgrp docker
        sudo apt-get install python3-pip
        ```

    * Install AWS-CLI
        ```
        curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
        sudo apt-get install unzip
        unzip awscli-bundle.zip
        ```
        
* Restart the EC2 machine
* Finally, in the Security Group of the EC2 Machine add a new rule to access the docker server 0.0.0.0/0 on port 8000

* Go to github repo settings>actions>runner>new self-hosted runner>linux
* Run the commands in EC2 instance created for web

* Add Github secrets:
    ```bash
	AWS_ACCESS_KEY_ID=
	AWS_SECRET_ACCESS_KEY=
	AWS_REGION= 
    AWS_S3_BUCKET=
	AWS_ECR_REPOSITORY_URI= 
	AWS_ECR_REPOSITORY_NAME= 
	MLFLOW_TRACKING_URI=
	```

