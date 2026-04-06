"""Generate diverse CloudFormation YAML gold configs from composable resource archetypes.

Each config picks 3-7 random archetypes, randomises names/CIDRs/regions/etc.,
and includes AWSTemplateFormatVersion, Description, Parameters, Resources,
Outputs, and cross-references via Ref, Fn::GetAtt, Fn::Sub, Fn::Join,
Fn::Select, and DependsOn.
Target: ~200 configs at 50-200 lines each.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import click
import yaml

# ---------------------------------------------------------------------------
# Randomisation helpers
# ---------------------------------------------------------------------------

REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "eu-west-1", "eu-west-2", "eu-central-1",
    "ap-southeast-1", "ap-northeast-1", "ca-central-1",
]

INSTANCE_TYPES = [
    "t3.micro", "t3.small", "t3.medium", "t3.large",
    "m5.large", "m5.xlarge", "c5.large", "r5.large",
]

AMIS = [
    "ami-0c55b159cbfafe1f0", "ami-0abcdef1234567890",
    "ami-08d70e59c07c61a3a", "ami-0742b4e673072066f",
    "ami-0d5eff06f840b45e9", "ami-09e67e426f25ce0d7",
]

ENVIRONMENTS = ["production", "staging", "development", "testing"]
DB_ENGINES = ["postgres", "mysql"]
DB_ENGINE_VERSIONS = {"postgres": ["14.9", "15.4", "16.1"], "mysql": ["8.0.35", "8.0.36"]}
RUNTIMES = ["python3.12", "python3.11", "nodejs20.x", "nodejs18.x"]
ALARM_METRICS = [
    ("CPUUtilization", "AWS/EC2", "Percent"),
    ("DatabaseConnections", "AWS/RDS", "Count"),
    ("4XXError", "AWS/ApplicationELB", "Count"),
    ("Duration", "AWS/Lambda", "Milliseconds"),
]

CIDR_SECOND_OCTETS = list(range(0, 255))


def _rand_name() -> str:
    adj = random.choice([
        "alpha", "beta", "gamma", "delta", "omega", "nova", "apex",
        "core", "flux", "edge", "prime", "nexus", "pulse", "arc",
        "stellar", "zen", "echo", "rapid", "titan", "vortex",
    ])
    noun = random.choice([
        "app", "svc", "api", "web", "data", "hub", "sys", "ops",
        "net", "cloud", "stack", "link", "node", "gate", "mesh",
    ])
    return f"{adj}-{noun}"


def _rand_cidr_16() -> str:
    return f"10.{random.choice(CIDR_SECOND_OCTETS)}.0.0/16"


def _rand_suffix(k: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def _rand_tags(name_val: str, env: str) -> list[dict]:
    return [
        {"Key": "Name", "Value": name_val},
        {"Key": "Environment", "Value": env},
        {"Key": "ManagedBy", "Value": "cloudformation"},
    ]


# ---------------------------------------------------------------------------
# Resource archetype functions — each mutates template dict in place
# ---------------------------------------------------------------------------

def archetype_vpc(template: dict, ctx: dict) -> dict:
    """Add VPC resource with DNS support."""
    project = ctx["project"]
    env = ctx["environment"]
    second = random.randint(0, 254)
    ctx["vpc_second"] = second
    cidr = f"10.{second}.0.0/16"
    ctx["vpc_cidr"] = cidr

    template["Parameters"]["VpcCidr"] = {
        "Type": "String",
        "Default": cidr,
        "Description": "CIDR block for the VPC",
    }
    template["Resources"]["VPC"] = {
        "Type": "AWS::EC2::VPC",
        "Properties": {
            "CidrBlock": {"Ref": "VpcCidr"},
            "EnableDnsSupport": True,
            "EnableDnsHostnames": True,
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-vpc"}, env),
        },
    }
    template["Outputs"]["VpcId"] = {
        "Description": "VPC ID",
        "Value": {"Ref": "VPC"},
    }
    return {"vpc": True}


def archetype_subnet(template: dict, ctx: dict) -> dict:
    """Add public and private subnets."""
    second = ctx.get("vpc_second", 0)
    env = ctx["environment"]
    pub1 = f"10.{second}.1.0/24"
    pub2 = f"10.{second}.2.0/24"
    priv1 = f"10.{second}.10.0/24"
    priv2 = f"10.{second}.11.0/24"

    template["Parameters"]["PublicSubnet1Cidr"] = {
        "Type": "String",
        "Default": pub1,
        "Description": "CIDR for public subnet 1",
    }
    template["Parameters"]["PublicSubnet2Cidr"] = {
        "Type": "String",
        "Default": pub2,
        "Description": "CIDR for public subnet 2",
    }

    def _azs():
        """Return a fresh Fn::GetAZs dict each time to avoid YAML anchors."""
        return {"Fn::GetAZs": {"Ref": "AWS::Region"}}

    template["Resources"]["PublicSubnet1"] = {
        "Type": "AWS::EC2::Subnet",
        "DependsOn": ["VPC"],
        "Properties": {
            "VpcId": {"Ref": "VPC"},
            "CidrBlock": {"Ref": "PublicSubnet1Cidr"},
            "AvailabilityZone": {"Fn::Select": [0, _azs()]},
            "MapPublicIpOnLaunch": True,
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-public-1"}, env),
        },
    }
    template["Resources"]["PublicSubnet2"] = {
        "Type": "AWS::EC2::Subnet",
        "DependsOn": ["VPC"],
        "Properties": {
            "VpcId": {"Ref": "VPC"},
            "CidrBlock": {"Ref": "PublicSubnet2Cidr"},
            "AvailabilityZone": {"Fn::Select": [1, _azs()]},
            "MapPublicIpOnLaunch": True,
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-public-2"}, env),
        },
    }
    template["Resources"]["PrivateSubnet1"] = {
        "Type": "AWS::EC2::Subnet",
        "DependsOn": ["VPC"],
        "Properties": {
            "VpcId": {"Ref": "VPC"},
            "CidrBlock": priv1,
            "AvailabilityZone": {"Fn::Select": [0, _azs()]},
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-private-1"}, env),
        },
    }
    template["Resources"]["PrivateSubnet2"] = {
        "Type": "AWS::EC2::Subnet",
        "DependsOn": ["VPC"],
        "Properties": {
            "VpcId": {"Ref": "VPC"},
            "CidrBlock": priv2,
            "AvailabilityZone": {"Fn::Select": [1, _azs()]},
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-private-2"}, env),
        },
    }

    template["Outputs"]["PublicSubnet1Id"] = {
        "Description": "Public Subnet 1 ID",
        "Value": {"Ref": "PublicSubnet1"},
    }
    return {"subnet": True}


def archetype_igw(template: dict, ctx: dict) -> dict:
    """Add Internet Gateway and VPC attachment."""
    env = ctx["environment"]
    template["Resources"]["InternetGateway"] = {
        "Type": "AWS::EC2::InternetGateway",
        "Properties": {
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-igw"}, env),
        },
    }
    template["Resources"]["VPCGatewayAttachment"] = {
        "Type": "AWS::EC2::VPCGatewayAttachment",
        "DependsOn": ["VPC", "InternetGateway"],
        "Properties": {
            "VpcId": {"Ref": "VPC"},
            "InternetGatewayId": {"Ref": "InternetGateway"},
        },
    }
    return {"igw": True}


def archetype_route_table(template: dict, ctx: dict) -> dict:
    """Add route table with default route through IGW."""
    env = ctx["environment"]
    template["Resources"]["PublicRouteTable"] = {
        "Type": "AWS::EC2::RouteTable",
        "DependsOn": ["VPC"],
        "Properties": {
            "VpcId": {"Ref": "VPC"},
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-public-rt"}, env),
        },
    }
    template["Resources"]["PublicRoute"] = {
        "Type": "AWS::EC2::Route",
        "DependsOn": ["PublicRouteTable", "VPCGatewayAttachment"],
        "Properties": {
            "RouteTableId": {"Ref": "PublicRouteTable"},
            "DestinationCidrBlock": "0.0.0.0/0",
            "GatewayId": {"Ref": "InternetGateway"},
        },
    }
    template["Resources"]["SubnetRouteTableAssoc1"] = {
        "Type": "AWS::EC2::SubnetRouteTableAssociation",
        "Properties": {
            "SubnetId": {"Ref": "PublicSubnet1"},
            "RouteTableId": {"Ref": "PublicRouteTable"},
        },
    }
    return {"route_table": True}


def archetype_security_group(template: dict, ctx: dict) -> dict:
    """Add security group with ingress rules (enables SECURITY injectors)."""
    env = ctx["environment"]
    ssh_cidr = f"10.{random.randint(0, 254)}.0.0/16"

    template["Resources"]["AppSecurityGroup"] = {
        "Type": "AWS::EC2::SecurityGroup",
        "DependsOn": ["VPC"],
        "Properties": {
            "GroupDescription": {"Fn::Sub": "Security group for ${ProjectName} application"},
            "VpcId": {"Ref": "VPC"},
            "SecurityGroupIngress": [
                {
                    "IpProtocol": "tcp",
                    "FromPort": 443,
                    "ToPort": 443,
                    "CidrIp": "0.0.0.0/0",
                    "Description": "HTTPS",
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "CidrIp": "0.0.0.0/0",
                    "Description": "HTTP",
                },
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "CidrIp": ssh_cidr,
                    "Description": "SSH from corporate",
                },
            ],
            "SecurityGroupEgress": [
                {
                    "IpProtocol": "-1",
                    "FromPort": 0,
                    "ToPort": 65535,
                    "CidrIp": "0.0.0.0/0",
                    "Description": "All outbound",
                },
            ],
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-app-sg"}, env),
        },
    }
    template["Outputs"]["SecurityGroupId"] = {
        "Description": "Application security group ID",
        "Value": {"Fn::GetAtt": ["AppSecurityGroup", "GroupId"]},
    }
    return {"security_group": True}


def archetype_ec2(template: dict, ctx: dict) -> dict:
    """Add EC2 instance with parameterised instance type and AMI."""
    itype = random.choice(INSTANCE_TYPES)
    ami = random.choice(AMIS)
    env = ctx["environment"]
    ctx["instance_type"] = itype
    ctx["ami"] = ami

    template["Parameters"]["InstanceType"] = {
        "Type": "String",
        "Default": itype,
        "Description": "EC2 instance type",
        "AllowedValues": INSTANCE_TYPES,
    }
    template["Parameters"]["AmiId"] = {
        "Type": "AWS::EC2::Image::Id",
        "Default": ami,
        "Description": "AMI ID for the EC2 instance",
    }

    template["Resources"]["AppInstance"] = {
        "Type": "AWS::EC2::Instance",
        "DependsOn": ["PublicSubnet1", "AppSecurityGroup"],
        "Properties": {
            "ImageId": {"Ref": "AmiId"},
            "InstanceType": {"Ref": "InstanceType"},
            "SubnetId": {"Ref": "PublicSubnet1"},
            "SecurityGroupIds": [{"Fn::GetAtt": ["AppSecurityGroup", "GroupId"]}],
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": random.choice([20, 30, 50]),
                        "VolumeType": "gp3",
                        "Encrypted": True,
                    },
                },
            ],
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-app-instance"}, env),
        },
    }
    template["Outputs"]["InstanceId"] = {
        "Description": "EC2 Instance ID",
        "Value": {"Ref": "AppInstance"},
    }
    template["Outputs"]["InstancePrivateIp"] = {
        "Description": "Private IP of the instance",
        "Value": {"Fn::GetAtt": ["AppInstance", "PrivateIp"]},
    }
    return {"ec2": True}


def archetype_s3(template: dict, ctx: dict) -> dict:
    """Add S3 bucket with encryption and versioning (enables CF_MISSING_ENCRYPTION)."""
    suffix = _rand_suffix()
    env = ctx["environment"]

    template["Resources"]["S3Bucket"] = {
        "Type": "AWS::S3::Bucket",
        "Properties": {
            "BucketName": {"Fn::Sub": "${ProjectName}-" + suffix},
            "VersioningConfiguration": {"Status": "Enabled"},
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [
                    {
                        "ServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "aws:kms",
                        },
                    },
                ],
            },
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-bucket"}, env),
        },
    }
    template["Outputs"]["S3BucketArn"] = {
        "Description": "S3 bucket ARN",
        "Value": {"Fn::GetAtt": ["S3Bucket", "Arn"]},
    }
    template["Outputs"]["S3BucketName"] = {
        "Description": "S3 bucket name",
        "Value": {"Ref": "S3Bucket"},
    }
    return {"s3": True}


def archetype_rds(template: dict, ctx: dict) -> dict:
    """Add RDS instance with encryption (enables CF_MISSING_ENCRYPTION)."""
    engine = random.choice(DB_ENGINES)
    version = random.choice(DB_ENGINE_VERSIONS[engine])
    instance_class = random.choice(["db.t3.micro", "db.t3.small", "db.t3.medium", "db.r5.large"])
    port = 5432 if engine == "postgres" else 3306
    env = ctx["environment"]

    template["Parameters"]["DBInstanceClass"] = {
        "Type": "String",
        "Default": instance_class,
        "Description": "RDS instance class",
    }
    template["Parameters"]["DBPassword"] = {
        "Type": "String",
        "NoEcho": True,
        "Default": "ChangeMe123!",
        "Description": "Database master password",
    }

    template["Resources"]["DBSubnetGroup"] = {
        "Type": "AWS::RDS::DBSubnetGroup",
        "Properties": {
            "DBSubnetGroupDescription": {"Fn::Sub": "${ProjectName} database subnet group"},
            "SubnetIds": [
                {"Ref": "PrivateSubnet1"},
                {"Ref": "PrivateSubnet2"},
            ],
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-db-subnet"}, env),
        },
    }

    template["Resources"]["DBSecurityGroup"] = {
        "Type": "AWS::EC2::SecurityGroup",
        "DependsOn": ["VPC"],
        "Properties": {
            "GroupDescription": {"Fn::Sub": "Database security group for ${ProjectName}"},
            "VpcId": {"Ref": "VPC"},
            "SecurityGroupIngress": [
                {
                    "IpProtocol": "tcp",
                    "FromPort": port,
                    "ToPort": port,
                    "SourceSecurityGroupId": {"Fn::GetAtt": ["AppSecurityGroup", "GroupId"]},
                    "Description": f"{engine.title()} from app tier",
                },
            ],
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-db-sg"}, env),
        },
    }

    template["Resources"]["RDSInstance"] = {
        "Type": "AWS::RDS::DBInstance",
        "DependsOn": ["DBSubnetGroup", "DBSecurityGroup"],
        "Properties": {
            "DBInstanceIdentifier": {"Fn::Sub": "${ProjectName}-db"},
            "Engine": engine,
            "EngineVersion": version,
            "DBInstanceClass": {"Ref": "DBInstanceClass"},
            "AllocatedStorage": random.choice([20, 50, 100]),
            "StorageEncrypted": True,
            "DBName": "appdb",
            "MasterUsername": "dbadmin",
            "MasterUserPassword": {"Ref": "DBPassword"},
            "DBSubnetGroupName": {"Ref": "DBSubnetGroup"},
            "VPCSecurityGroups": [{"Fn::GetAtt": ["DBSecurityGroup", "GroupId"]}],
            "MultiAZ": random.choice([True, False]),
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-db"}, env),
        },
    }

    template["Outputs"]["RDSEndpoint"] = {
        "Description": "RDS endpoint address",
        "Value": {"Fn::GetAtt": ["RDSInstance", "Endpoint.Address"]},
    }
    return {"rds": True}


def archetype_lambda_fn(template: dict, ctx: dict) -> dict:
    """Add Lambda function with log group."""
    runtime = random.choice(RUNTIMES)
    timeout = random.choice([30, 60, 120, 300])
    mem = random.choice([128, 256, 512, 1024])
    env = ctx["environment"]

    template["Resources"]["LambdaFunction"] = {
        "Type": "AWS::Lambda::Function",
        "DependsOn": ["LambdaExecutionRole"],
        "Properties": {
            "FunctionName": {"Fn::Sub": "${ProjectName}-handler"},
            "Runtime": runtime,
            "Handler": "index.handler",
            "Role": {"Fn::GetAtt": ["LambdaExecutionRole", "Arn"]},
            "Timeout": timeout,
            "MemorySize": mem,
            "Code": {
                "ZipFile": {"Fn::Join": ["\n", [
                    "import json",
                    "def handler(event, context):",
                    "    return {'statusCode': 200, 'body': json.dumps('OK')}",
                ]]},
            },
            "Environment": {
                "Variables": {
                    "ENVIRONMENT": {"Ref": "Environment"},
                    "PROJECT": {"Ref": "ProjectName"},
                },
            },
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-lambda"}, env),
        },
    }

    template["Resources"]["LambdaLogGroup"] = {
        "Type": "AWS::Logs::LogGroup",
        "Properties": {
            "LogGroupName": {"Fn::Sub": "/aws/lambda/${ProjectName}-handler"},
            "RetentionInDays": random.choice([7, 14, 30, 90]),
        },
    }

    template["Outputs"]["LambdaArn"] = {
        "Description": "Lambda function ARN",
        "Value": {"Fn::GetAtt": ["LambdaFunction", "Arn"]},
    }
    return {"lambda_fn": True}


def archetype_iam_role(template: dict, ctx: dict) -> dict:
    """Add IAM execution role for Lambda."""
    env = ctx["environment"]

    template["Resources"]["LambdaExecutionRole"] = {
        "Type": "AWS::IAM::Role",
        "Properties": {
            "RoleName": {"Fn::Sub": "${ProjectName}-lambda-role"},
            "AssumeRolePolicyDocument": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    },
                ],
            },
            "ManagedPolicyArns": [
                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            ],
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-lambda-role"}, env),
        },
    }
    template["Outputs"]["LambdaRoleArn"] = {
        "Description": "Lambda execution role ARN",
        "Value": {"Fn::GetAtt": ["LambdaExecutionRole", "Arn"]},
    }
    return {"iam_role": True}


def archetype_dynamodb(template: dict, ctx: dict) -> dict:
    """Add DynamoDB table with SSE (enables CF_MISSING_ENCRYPTION)."""
    env = ctx["environment"]
    read_cap = random.choice([5, 10, 25])
    write_cap = random.choice([5, 10, 25])

    template["Resources"]["DynamoDBTable"] = {
        "Type": "AWS::DynamoDB::Table",
        "Properties": {
            "TableName": {"Fn::Sub": "${ProjectName}-table"},
            "AttributeDefinitions": [
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": read_cap,
                "WriteCapacityUnits": write_cap,
            },
            "SSESpecification": {"SSEEnabled": True},
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-dynamodb"}, env),
        },
    }
    template["Outputs"]["DynamoDBTableName"] = {
        "Description": "DynamoDB table name",
        "Value": {"Ref": "DynamoDBTable"},
    }
    template["Outputs"]["DynamoDBTableArn"] = {
        "Description": "DynamoDB table ARN",
        "Value": {"Fn::GetAtt": ["DynamoDBTable", "Arn"]},
    }
    return {"dynamodb": True}


def archetype_sns_sqs(template: dict, ctx: dict) -> dict:
    """Add SNS topic and SQS queue with subscription."""
    env = ctx["environment"]

    template["Resources"]["AlertsTopic"] = {
        "Type": "AWS::SNS::Topic",
        "Properties": {
            "TopicName": {"Fn::Sub": "${ProjectName}-alerts"},
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-alerts"}, env),
        },
    }
    template["Resources"]["ProcessingQueue"] = {
        "Type": "AWS::SQS::Queue",
        "Properties": {
            "QueueName": {"Fn::Sub": "${ProjectName}-processing"},
            "VisibilityTimeout": random.choice([30, 60, 120, 300]),
            "MessageRetentionPeriod": random.choice([345600, 604800, 1209600]),
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-queue"}, env),
        },
    }
    template["Resources"]["SQSSubscription"] = {
        "Type": "AWS::SNS::Subscription",
        "DependsOn": ["AlertsTopic", "ProcessingQueue"],
        "Properties": {
            "TopicArn": {"Ref": "AlertsTopic"},
            "Protocol": "sqs",
            "Endpoint": {"Fn::GetAtt": ["ProcessingQueue", "Arn"]},
        },
    }
    template["Outputs"]["SNSTopicArn"] = {
        "Description": "SNS topic ARN",
        "Value": {"Ref": "AlertsTopic"},
    }
    template["Outputs"]["SQSQueueUrl"] = {
        "Description": "SQS queue URL",
        "Value": {"Ref": "ProcessingQueue"},
    }
    return {"sns_sqs": True}


def archetype_alb(template: dict, ctx: dict) -> dict:
    """Add Application Load Balancer with target group and listener."""
    env = ctx["environment"]

    template["Resources"]["ApplicationLB"] = {
        "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
        "DependsOn": ["PublicSubnet1", "PublicSubnet2", "AppSecurityGroup"],
        "Properties": {
            "Name": {"Fn::Sub": "${ProjectName}-alb"},
            "Scheme": "internet-facing",
            "Type": "application",
            "SecurityGroups": [{"Fn::GetAtt": ["AppSecurityGroup", "GroupId"]}],
            "Subnets": [
                {"Ref": "PublicSubnet1"},
                {"Ref": "PublicSubnet2"},
            ],
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-alb"}, env),
        },
    }
    template["Resources"]["ALBTargetGroup"] = {
        "Type": "AWS::ElasticLoadBalancingV2::TargetGroup",
        "DependsOn": ["VPC"],
        "Properties": {
            "Name": {"Fn::Sub": "${ProjectName}-tg"},
            "Port": 80,
            "Protocol": "HTTP",
            "VpcId": {"Ref": "VPC"},
            "HealthCheckPath": "/health",
            "HealthyThresholdCount": 3,
            "UnhealthyThresholdCount": 3,
            "HealthCheckTimeoutSeconds": 5,
            "HealthCheckIntervalSeconds": 30,
            "Tags": _rand_tags({"Fn::Sub": "${ProjectName}-tg"}, env),
        },
    }
    template["Resources"]["ALBListener"] = {
        "Type": "AWS::ElasticLoadBalancingV2::Listener",
        "DependsOn": ["ApplicationLB", "ALBTargetGroup"],
        "Properties": {
            "LoadBalancerArn": {"Ref": "ApplicationLB"},
            "Port": 80,
            "Protocol": "HTTP",
            "DefaultActions": [
                {
                    "Type": "forward",
                    "TargetGroupArn": {"Ref": "ALBTargetGroup"},
                },
            ],
        },
    }
    template["Outputs"]["ALBDnsName"] = {
        "Description": "ALB DNS name",
        "Value": {"Fn::GetAtt": ["ApplicationLB", "DNSName"]},
    }
    template["Outputs"]["ALBFullName"] = {
        "Description": "ALB full name",
        "Value": {"Fn::Join": ["-", [
            {"Ref": "ProjectName"},
            "alb",
            {"Ref": "Environment"},
        ]]},
    }
    return {"alb": True}


def archetype_cloudwatch(template: dict, ctx: dict) -> dict:
    """Add CloudWatch alarm (optionally linked to SNS)."""
    metric, namespace, unit = random.choice(ALARM_METRICS)
    threshold = random.choice([70, 80, 90, 95, 100, 500, 1000])
    env = ctx["environment"]

    alarm_props = {
        "AlarmName": {"Fn::Sub": "${ProjectName}-" + metric.lower() + "-alarm"},
        "AlarmDescription": f"Alarm when {metric} exceeds {threshold}",
        "ComparisonOperator": "GreaterThanThreshold",
        "EvaluationPeriods": random.choice([1, 2, 3]),
        "MetricName": metric,
        "Namespace": namespace,
        "Period": random.choice([60, 120, 300]),
        "Statistic": "Average",
        "Threshold": threshold,
    }

    if ctx.get("_has_sns_sqs"):
        alarm_props["AlarmActions"] = [{"Ref": "AlertsTopic"}]

    template["Resources"]["CloudWatchAlarm"] = {
        "Type": "AWS::CloudWatch::Alarm",
        "Properties": alarm_props,
    }
    template["Outputs"]["AlarmArn"] = {
        "Description": "CloudWatch alarm ARN",
        "Value": {"Fn::GetAtt": ["CloudWatchAlarm", "Arn"]},
    }
    return {"cloudwatch": True}


# ---------------------------------------------------------------------------
# Archetype registry with dependency info
# ---------------------------------------------------------------------------

# (function, requires, name)
ARCHETYPES = [
    (archetype_vpc,            set(),                                     "vpc"),
    (archetype_subnet,         {"vpc"},                                   "subnet"),
    (archetype_igw,            {"vpc"},                                   "igw"),
    (archetype_route_table,    {"vpc", "igw", "subnet"},                  "route_table"),
    (archetype_security_group, {"vpc"},                                   "security_group"),
    (archetype_ec2,            {"subnet", "security_group"},              "ec2"),
    (archetype_s3,             set(),                                     "s3"),
    (archetype_rds,            {"vpc", "subnet", "security_group"},       "rds"),
    (archetype_lambda_fn,      {"iam_role"},                              "lambda_fn"),
    (archetype_iam_role,       set(),                                     "iam_role"),
    (archetype_dynamodb,       set(),                                     "dynamodb"),
    (archetype_sns_sqs,        set(),                                     "sns_sqs"),
    (archetype_alb,            {"vpc", "subnet", "security_group"},       "alb"),
    (archetype_cloudwatch,     set(),                                     "cloudwatch"),
]


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _build_config(seed: int, pick_count: int | None = None) -> str:
    """Build a single CloudFormation YAML config from random archetypes."""
    rng = random.Random(seed)
    random.seed(seed)

    project = _rand_name()
    region = rng.choice(REGIONS)
    env = rng.choice(ENVIRONMENTS)

    if pick_count is None:
        pick_count = rng.randint(3, 7)

    # Select archetypes respecting dependencies
    available = list(ARCHETYPES)
    rng.shuffle(available)
    selected: list[tuple] = []
    selected_names: set[str] = set()

    for func, deps, name in available:
        if len(selected) >= pick_count:
            break
        # Add any missing dependencies first
        missing = deps - selected_names
        if missing:
            for mfunc, mdeps, mname in ARCHETYPES:
                if mname in missing and mname not in selected_names:
                    # Recursively add deps of the dep
                    for dfunc, ddeps, dname in ARCHETYPES:
                        if dname in mdeps and dname not in selected_names:
                            selected.append((dfunc, ddeps, dname))
                            selected_names.add(dname)
                    selected.append((mfunc, mdeps, mname))
                    selected_names.add(mname)
        selected.append((func, deps, name))
        selected_names.add(name)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for item in selected:
        if item[2] not in seen:
            seen.add(item[2])
            deduped.append(item)
    selected = deduped

    # Sort by dependency order
    order = [name for _, _, name in ARCHETYPES]
    selected.sort(key=lambda x: order.index(x[2]))

    # Build context
    ctx = {
        "project": project,
        "region": region,
        "environment": env,
        "_has_sns_sqs": "sns_sqs" in selected_names,
    }

    # Initialise template structure
    template: dict = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"Infrastructure stack for {project} ({env}) in {region}",
        "Parameters": {
            "ProjectName": {
                "Type": "String",
                "Default": project,
                "Description": "Project name used as prefix for resources",
            },
            "Environment": {
                "Type": "String",
                "Default": env,
                "Description": "Deployment environment",
                "AllowedValues": ENVIRONMENTS,
            },
        },
        "Resources": {},
        "Outputs": {},
    }

    # Generate each archetype
    for func, deps, name in selected:
        func(template, ctx)

    return yaml.dump(template, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--count", "-n", default=200, help="Number of configs to generate")
@click.option(
    "--output-dir", "-o",
    default="data/gold/cloudformation/generated",
    help="Output directory",
)
@click.option("--seed", "-s", default=1000, help="Base random seed")
def main(count: int, output_dir: str, seed: int):
    """Generate diverse CloudFormation YAML gold configurations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        config = _build_config(seed + i)
        filename = f"config_{i:04d}.yaml"
        (out / filename).write_text(config)

    click.echo(f"Generated {count} gold configs in {out}/")


if __name__ == "__main__":
    main()
