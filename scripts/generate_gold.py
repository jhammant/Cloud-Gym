"""Generate diverse Terraform gold configs from composable resource archetypes.

Each config picks 3-7 random archetypes, randomises names/CIDRs/regions/etc.,
and includes terraform/provider blocks, variables, outputs, and cross-references.
Target: ~150 configs at 50-200 lines each.
"""

from __future__ import annotations

import random
import string
import textwrap
from pathlib import Path

import click

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


def _rand_subnet_cidrs(vpc_second: int, count: int, offset: int = 0) -> list[str]:
    return [f"10.{vpc_second}.{offset + i}.0/24" for i in range(count)]


def _rand_tag_block(name_val: str, extra: dict[str, str] | None = None) -> str:
    tags = {"Name": name_val, "Environment": random.choice(ENVIRONMENTS), "ManagedBy": "terraform"}
    if extra:
        tags.update(extra)
    lines = [f'    {k:16s} = "{v}"' for k, v in tags.items()]
    return "  tags = {\n" + "\n".join(lines) + "\n  }"


# ---------------------------------------------------------------------------
# Resource archetype functions — each returns (hcl_text, exports)
# exports is a dict of logical names produced (for cross-referencing)
# ---------------------------------------------------------------------------

def archetype_vpc(ctx: dict) -> tuple[str, dict]:
    name = ctx["project"]
    second = random.randint(0, 254)
    ctx["vpc_second"] = second
    cidr = f"10.{second}.0.0/16"
    ctx["vpc_cidr"] = cidr
    return textwrap.dedent(f"""\
        resource "aws_vpc" "main" {{
          cidr_block           = var.vpc_cidr
          enable_dns_support   = true
          enable_dns_hostnames = true

        {_rand_tag_block(f"${{var.project_name}}-vpc")}
        }}
    """), {"vpc": True}


def archetype_subnet(ctx: dict) -> tuple[str, dict]:
    second = ctx.get("vpc_second", 0)
    pub_cidrs = _rand_subnet_cidrs(second, 2, offset=1)
    priv_cidrs = _rand_subnet_cidrs(second, 2, offset=10)
    ctx["pub_cidrs"] = pub_cidrs
    ctx["priv_cidrs"] = priv_cidrs
    return textwrap.dedent(f"""\
        variable "public_subnet_cidrs" {{
          description = "CIDR blocks for public subnets"
          type        = list(string)
          default     = {json_list(pub_cidrs)}
        }}

        variable "private_subnet_cidrs" {{
          description = "CIDR blocks for private subnets"
          type        = list(string)
          default     = {json_list(priv_cidrs)}
        }}

        data "aws_availability_zones" "available" {{
          state = "available"
        }}

        resource "aws_subnet" "public" {{
          count                   = length(var.public_subnet_cidrs)
          vpc_id                  = aws_vpc.main.id
          cidr_block              = var.public_subnet_cidrs[count.index]
          availability_zone       = data.aws_availability_zones.available.names[count.index]
          map_public_ip_on_launch = true

        {_rand_tag_block('${var.project_name}-public-${count.index + 1}', {"Tier": "public"})}
        }}

        resource "aws_subnet" "private" {{
          count             = length(var.private_subnet_cidrs)
          vpc_id            = aws_vpc.main.id
          cidr_block        = var.private_subnet_cidrs[count.index]
          availability_zone = data.aws_availability_zones.available.names[count.index]

        {_rand_tag_block('${var.project_name}-private-${count.index + 1}', {"Tier": "private"})}
        }}
    """), {"subnet": True}


def archetype_igw(ctx: dict) -> tuple[str, dict]:
    return textwrap.dedent(f"""\
        resource "aws_internet_gateway" "main" {{
          vpc_id = aws_vpc.main.id

        {_rand_tag_block('${var.project_name}-igw')}
        }}
    """), {"igw": True}


def archetype_route_table(ctx: dict) -> tuple[str, dict]:
    return textwrap.dedent("""\
        resource "aws_route_table" "public" {
          vpc_id = aws_vpc.main.id

          route {
            cidr_block = "0.0.0.0/0"
            gateway_id = aws_internet_gateway.main.id
          }

          tags = {
            Name = "${var.project_name}-public-rt"
          }
        }

        resource "aws_route_table_association" "public" {
          count          = length(aws_subnet.public)
          subnet_id      = aws_subnet.public[count.index].id
          route_table_id = aws_route_table.public.id
        }
    """), {"route_table": True}


def archetype_security_group(ctx: dict) -> tuple[str, dict]:
    ssh_cidr = f"10.{random.randint(0,254)}.0.0/16"
    return textwrap.dedent(f"""\
        resource "aws_security_group" "app" {{
          name        = "${{var.project_name}}-app-sg"
          description = "Security group for application tier"
          vpc_id      = aws_vpc.main.id

          ingress {{
            description = "HTTPS"
            from_port   = 443
            to_port     = 443
            protocol    = "tcp"
            cidr_blocks = ["0.0.0.0/0"]
          }}

          ingress {{
            description = "HTTP"
            from_port   = 80
            to_port     = 80
            protocol    = "tcp"
            cidr_blocks = ["0.0.0.0/0"]
          }}

          ingress {{
            description = "SSH from corporate"
            from_port   = 22
            to_port     = 22
            protocol    = "tcp"
            cidr_blocks = ["{ssh_cidr}"]
          }}

          egress {{
            description = "All outbound"
            from_port   = 0
            to_port     = 0
            protocol    = "-1"
            cidr_blocks = ["0.0.0.0/0"]
          }}

        {_rand_tag_block('${var.project_name}-app-sg')}
        }}
    """), {"security_group": True}


def archetype_ec2(ctx: dict) -> tuple[str, dict]:
    itype = random.choice(INSTANCE_TYPES)
    ami = random.choice(AMIS)
    ctx["instance_type"] = itype
    ctx["ami"] = ami
    return textwrap.dedent(f"""\
        variable "instance_type" {{
          description = "EC2 instance type"
          type        = string
          default     = "{itype}"
        }}

        variable "ami_id" {{
          description = "AMI ID for the instance"
          type        = string
          default     = "{ami}"
        }}

        resource "aws_instance" "app" {{
          ami                    = var.ami_id
          instance_type          = var.instance_type
          subnet_id              = aws_subnet.public[0].id
          vpc_security_group_ids = [aws_security_group.app.id]

          root_block_device {{
            volume_type = "gp3"
            volume_size = {random.choice([20, 30, 50])}
            encrypted   = true
          }}

          metadata_options {{
            http_endpoint = "enabled"
            http_tokens   = "required"
          }}

          monitoring = true

        {_rand_tag_block('${var.project_name}-app', {"Role": "application"})}
        }}

        output "instance_id" {{
          description = "ID of the application instance"
          value       = aws_instance.app.id
        }}
    """), {"ec2": True}


def archetype_ebs(ctx: dict) -> tuple[str, dict]:
    size = random.choice([50, 100, 200, 500])
    return textwrap.dedent(f"""\
        resource "aws_ebs_volume" "data" {{
          availability_zone = aws_instance.app.availability_zone
          size              = {size}
          type              = "gp3"
          encrypted         = true
          iops              = 3000
          throughput        = 125

        {_rand_tag_block('${var.project_name}-data-vol')}
        }}

        resource "aws_volume_attachment" "data" {{
          device_name = "/dev/xvdf"
          volume_id   = aws_ebs_volume.data.id
          instance_id = aws_instance.app.id
        }}
    """), {"ebs": True}


def archetype_s3(ctx: dict) -> tuple[str, dict]:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return textwrap.dedent(f"""\
        resource "aws_s3_bucket" "main" {{
          bucket = "${{var.project_name}}-{suffix}"

        {_rand_tag_block('${var.project_name}-bucket')}
        }}

        resource "aws_s3_bucket_versioning" "main" {{
          bucket = aws_s3_bucket.main.id

          versioning_configuration {{
            status = "Enabled"
          }}
        }}

        resource "aws_s3_bucket_server_side_encryption_configuration" "main" {{
          bucket = aws_s3_bucket.main.id

          rule {{
            apply_server_side_encryption_by_default {{
              sse_algorithm = "aws:kms"
            }}
          }}
        }}

        resource "aws_s3_bucket_public_access_block" "main" {{
          bucket = aws_s3_bucket.main.id

          block_public_acls       = true
          block_public_policy     = true
          ignore_public_acls      = true
          restrict_public_buckets = true
        }}

        output "s3_bucket_arn" {{
          description = "ARN of the S3 bucket"
          value       = aws_s3_bucket.main.arn
        }}
    """), {"s3": True}


def archetype_rds(ctx: dict) -> tuple[str, dict]:
    engine = random.choice(DB_ENGINES)
    version = random.choice(DB_ENGINE_VERSIONS[engine])
    instance_class = random.choice(["db.t3.micro", "db.t3.small", "db.t3.medium", "db.r5.large"])
    port = 5432 if engine == "postgres" else 3306
    return textwrap.dedent(f"""\
        resource "aws_db_subnet_group" "main" {{
          name       = "${{var.project_name}}-db-subnet"
          subnet_ids = aws_subnet.private[*].id

        {_rand_tag_block('${var.project_name}-db-subnet')}
        }}

        resource "aws_security_group" "database" {{
          name        = "${{var.project_name}}-db-sg"
          description = "Security group for database"
          vpc_id      = aws_vpc.main.id

          ingress {{
            description     = "{engine.title()} from app tier"
            from_port       = {port}
            to_port         = {port}
            protocol        = "tcp"
            security_groups = [aws_security_group.app.id]
          }}

          egress {{
            from_port   = 0
            to_port     = 0
            protocol    = "-1"
            cidr_blocks = ["0.0.0.0/0"]
          }}

        {_rand_tag_block('${var.project_name}-db-sg')}
        }}

        resource "aws_db_instance" "main" {{
          identifier           = "${{var.project_name}}-db"
          engine               = "{engine}"
          engine_version       = "{version}"
          instance_class       = "{instance_class}"
          allocated_storage    = {random.choice([20, 50, 100])}
          storage_encrypted    = true
          db_name              = "appdb"
          username             = "dbadmin"
          password             = "ChangeMe123!"
          db_subnet_group_name = aws_db_subnet_group.main.name
          vpc_security_group_ids = [aws_security_group.database.id]
          skip_final_snapshot  = true
          multi_az             = {str(random.choice([True, False])).lower()}

        {_rand_tag_block('${var.project_name}-db')}
        }}

        output "rds_endpoint" {{
          description = "RDS instance endpoint"
          value       = aws_db_instance.main.endpoint
        }}
    """), {"rds": True}


def archetype_lambda(ctx: dict) -> tuple[str, dict]:
    runtime = random.choice(RUNTIMES)
    timeout = random.choice([30, 60, 120, 300])
    mem = random.choice([128, 256, 512, 1024])
    return textwrap.dedent(f"""\
        resource "aws_lambda_function" "main" {{
          function_name = "${{var.project_name}}-handler"
          role          = aws_iam_role.lambda_exec.arn
          handler       = "index.handler"
          runtime       = "{runtime}"
          timeout       = {timeout}
          memory_size   = {mem}
          filename      = "lambda.zip"

          environment {{
            variables = {{
              ENVIRONMENT = var.environment
            }}
          }}

        {_rand_tag_block('${var.project_name}-lambda')}
        }}

        resource "aws_cloudwatch_log_group" "lambda" {{
          name              = "/aws/lambda/${{var.project_name}}-handler"
          retention_in_days = {random.choice([7, 14, 30, 90])}

        {_rand_tag_block('${var.project_name}-lambda-logs')}
        }}

        output "lambda_arn" {{
          description = "ARN of the Lambda function"
          value       = aws_lambda_function.main.arn
        }}
    """), {"lambda": True}


def archetype_iam_role(ctx: dict) -> tuple[str, dict]:
    return textwrap.dedent("""\
        resource "aws_iam_role" "lambda_exec" {
          name = "${var.project_name}-lambda-role"

          assume_role_policy = jsonencode({
            Version = "2012-10-17"
            Statement = [{
              Action = "sts:AssumeRole"
              Effect = "Allow"
              Principal = {
                Service = "lambda.amazonaws.com"
              }
            }]
          })

          tags = {
            Name = "${var.project_name}-lambda-role"
          }
        }

        resource "aws_iam_role_policy_attachment" "lambda_basic" {
          role       = aws_iam_role.lambda_exec.name
          policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        }
    """), {"iam_role": True}


def archetype_alb(ctx: dict) -> tuple[str, dict]:
    return textwrap.dedent(f"""\
        resource "aws_lb" "main" {{
          name               = "${{var.project_name}}-alb"
          internal           = false
          load_balancer_type = "application"
          security_groups    = [aws_security_group.app.id]
          subnets            = aws_subnet.public[*].id

        {_rand_tag_block('${var.project_name}-alb')}
        }}

        resource "aws_lb_target_group" "app" {{
          name     = "${{var.project_name}}-tg"
          port     = 80
          protocol = "HTTP"
          vpc_id   = aws_vpc.main.id

          health_check {{
            path                = "/health"
            healthy_threshold   = 3
            unhealthy_threshold = 3
            timeout             = 5
            interval            = 30
          }}

        {_rand_tag_block('${var.project_name}-tg')}
        }}

        resource "aws_lb_listener" "http" {{
          load_balancer_arn = aws_lb.main.arn
          port              = 80
          protocol          = "HTTP"

          default_action {{
            type             = "forward"
            target_group_arn = aws_lb_target_group.app.arn
          }}
        }}

        output "alb_dns_name" {{
          description = "DNS name of the ALB"
          value       = aws_lb.main.dns_name
        }}
    """), {"alb": True}


def archetype_asg(ctx: dict) -> tuple[str, dict]:
    ami = ctx.get("ami", random.choice(AMIS))
    itype = ctx.get("instance_type", random.choice(INSTANCE_TYPES))
    return textwrap.dedent(f"""\
        resource "aws_launch_template" "app" {{
          name_prefix   = "${{var.project_name}}-lt-"
          image_id      = "{ami}"
          instance_type = "{itype}"

          vpc_security_group_ids = [aws_security_group.app.id]

          metadata_options {{
            http_endpoint = "enabled"
            http_tokens   = "required"
          }}

          block_device_mappings {{
            device_name = "/dev/xvda"
            ebs {{
              volume_size = 30
              volume_type = "gp3"
              encrypted   = true
            }}
          }}

          tag_specifications {{
            resource_type = "instance"
            tags = {{
              Name        = "${{var.project_name}}-asg-instance"
              Environment = var.environment
            }}
          }}
        }}

        resource "aws_autoscaling_group" "app" {{
          name                = "${{var.project_name}}-asg"
          min_size            = {random.choice([1, 2])}
          max_size            = {random.choice([4, 6, 10])}
          desired_capacity    = {random.choice([2, 3])}
          vpc_zone_identifier = aws_subnet.public[*].id
          {"target_group_arns  = [aws_lb_target_group.app.arn]" if ctx.get("_has_alb") else ""}

          launch_template {{
            id      = aws_launch_template.app.id
            version = "$Latest"
          }}

          tag {{
            key                 = "Name"
            value               = "${{var.project_name}}-asg"
            propagate_at_launch = true
          }}
        }}
    """), {"asg": True}


def archetype_cloudwatch(ctx: dict) -> tuple[str, dict]:
    metric, namespace, unit = random.choice(ALARM_METRICS)
    threshold = random.choice([70, 80, 90, 95, 100, 500, 1000])
    return textwrap.dedent(f"""\
        resource "aws_cloudwatch_metric_alarm" "main" {{
          alarm_name          = "${{var.project_name}}-{metric.lower()}-alarm"
          comparison_operator = "GreaterThanThreshold"
          evaluation_periods  = {random.choice([1, 2, 3])}
          metric_name         = "{metric}"
          namespace           = "{namespace}"
          period              = {random.choice([60, 120, 300])}
          statistic           = "Average"
          threshold           = {threshold}
          alarm_description   = "Alarm when {metric} exceeds {threshold}"
          {"alarm_actions      = [aws_sns_topic.alerts.arn]" if ctx.get("_has_sns") else ""}

        {_rand_tag_block('${var.project_name}-alarm')}
        }}
    """), {"cloudwatch": True}


def archetype_sns(ctx: dict) -> tuple[str, dict]:
    return textwrap.dedent(f"""\
        resource "aws_sns_topic" "alerts" {{
          name = "${{var.project_name}}-alerts"

        {_rand_tag_block('${var.project_name}-alerts')}
        }}

        resource "aws_sns_topic_subscription" "email" {{
          topic_arn = aws_sns_topic.alerts.arn
          protocol  = "email"
          endpoint  = "alerts@example.com"
        }}

        output "sns_topic_arn" {{
          description = "ARN of the SNS alerts topic"
          value       = aws_sns_topic.alerts.arn
        }}
    """), {"sns": True}


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
    (archetype_ebs,            {"ec2"},                                   "ebs"),
    (archetype_s3,             set(),                                     "s3"),
    (archetype_rds,            {"vpc", "subnet", "security_group"},       "rds"),
    (archetype_lambda,         {"iam_role"},                              "lambda"),
    (archetype_iam_role,       set(),                                     "iam_role"),
    (archetype_alb,            {"vpc", "subnet", "security_group"},       "alb"),
    (archetype_asg,            {"subnet", "security_group"},              "asg"),
    (archetype_cloudwatch,     set(),                                     "cloudwatch"),
    (archetype_sns,            set(),                                     "sns"),
]


def json_list(items: list[str]) -> str:
    inner = ", ".join(f'"{i}"' for i in items)
    return f"[{inner}]"


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _build_config(seed: int, pick_count: int | None = None) -> str:
    """Build a single TF config from random archetypes."""
    rng = random.Random(seed)
    random.seed(seed)  # also set module-level for helper funcs

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
    seen = set()
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
        "_has_alb": "alb" in selected_names,
        "_has_sns": "sns" in selected_names,
    }

    # Header
    second = rng.randint(0, 254)
    ctx["vpc_second"] = second
    ctx["vpc_cidr"] = f"10.{second}.0.0/16"

    parts = [
        textwrap.dedent(f"""\
            terraform {{
              required_version = ">= 1.5.0"

              required_providers {{
                aws = {{
                  source  = "hashicorp/aws"
                  version = "~> 5.0"
                }}
              }}
            }}

            provider "aws" {{
              region = var.aws_region
            }}

            variable "aws_region" {{
              description = "AWS region"
              type        = string
              default     = "{region}"
            }}

            variable "project_name" {{
              description = "Project name prefix"
              type        = string
              default     = "{project}"
            }}

            variable "environment" {{
              description = "Environment name"
              type        = string
              default     = "{env}"
            }}
        """),
    ]

    # Add VPC CIDR variable if VPC is selected
    if "vpc" in selected_names:
        parts.append(textwrap.dedent(f"""\
            variable "vpc_cidr" {{
              description = "CIDR block for the VPC"
              type        = string
              default     = "{ctx['vpc_cidr']}"
            }}
        """))

    # Generate each archetype
    for func, deps, name in selected:
        hcl, exports = func(ctx)
        parts.append(hcl)

    # Add VPC output if VPC present
    if "vpc" in selected_names:
        parts.append(textwrap.dedent("""\
            output "vpc_id" {
              description = "ID of the VPC"
              value       = aws_vpc.main.id
            }
        """))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--count", "-n", default=150, help="Number of configs to generate")
@click.option(
    "--output-dir", "-o",
    default="data/gold/terraform/generated",
    help="Output directory",
)
@click.option("--seed", "-s", default=42, help="Base random seed")
def main(count: int, output_dir: str, seed: int):
    """Generate diverse Terraform gold configurations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        config = _build_config(seed + i)
        filename = f"config_{i:04d}.tf"
        (out / filename).write_text(config)

    click.echo(f"Generated {count} gold configs in {out}/")


if __name__ == "__main__":
    main()
