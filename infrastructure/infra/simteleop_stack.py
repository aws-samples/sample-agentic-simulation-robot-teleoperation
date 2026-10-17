"""One stack: network, bucket, IAM and a GPU instance for the simulated teleoperation platform.

Adapted from the OpenArm data-collection stack (same account patterns, same security posture) with
one simplification: there is no private container registry. The simulator image is the NVIDIA
Sim-to-Real SO-101 workshop's own Dockerfile on top of the public ``nvcr.io/nvidia/isaac-lab``
image, built on the instance at first boot, so nothing has to be seeded before the box is useful.
"""
from __future__ import annotations

from typing import Optional, Sequence

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct


class SimTeleopPlatformStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        instance_type: str = "g7e.4xlarge",
        ami_id: Optional[str] = None,
        # Pinned to the last Deep Learning Base GPU AMI build with the NVIDIA 580 driver (580.159.04).
        # Builds from 20260717 on ship the 595 driver, on which Isaac Sim 5.1 (the workshop's Isaac Lab
        # 2.3.2 image) segfaults at startup in the RTX scene renderer -- measured 2026-10-08 on a fresh
        # g6e.4xlarge with 595.91.07; the same image runs on 580.178.04 / 580.159.04. Override with
        # -c amiNameFilter='... *' or -c amiId=... once the workshop moves to Isaac Lab 3.0 / Isaac Sim 6.
        ami_name_filter: str = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260623",
        root_volume_gb: int = 400,
        availability_zone: Optional[str] = None,
        max_azs: int = 3,
        dcv_allowed_cidr: Optional[str] = None,
        operator_principal_arns: Sequence[str] = (),
        allow_account_root_operator: bool = False,
        key_name: Optional[str] = None,
        workshop_repo: str = "https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git",
        workshop_commit: str = "ce807d99724cb65671abec01f908a2fcb4a6eab7",
        agent_model_id: str = "global.openai.gpt-6-astra",
        enable_agent_bedrock: bool = True,
        retain_data_on_destroy: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        Tags.of(self).add("project", "sim-teleop-platform")

        # ------------------------------------------------------------------ network
        # Public subnets in every AZ, no NAT: DCV needs a routable address, and GPU capacity is per
        # AZ, so a shortage is a redeploy with -c availabilityZone=... rather than a network change.
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=max_azs,
            nat_gateways=0,
            ip_addresses=ec2.IpAddresses.cidr("10.30.0.0/16"),
            subnet_configuration=[ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)],
        )

        # ------------------------------------------------------------------ storage
        bucket = s3.Bucket(
            self,
            "ExchangeBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            # Versioned: the `code/` prefix is executed as root at boot, so every change to it must be
            # recoverable and attributable; also undoes an accidental dataset delete (security review F-1).
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN if retain_data_on_destroy else RemovalPolicy.DESTROY,
            auto_delete_objects=not retain_data_on_destroy,
            lifecycle_rules=[s3.LifecycleRule(id="expire-run-logs", prefix="runs/", expiration=Duration.days(90), abort_incomplete_multipart_upload_after=Duration.days(7))],
        )

        # The repository's code goes to the bucket; user data pulls it. prune=False so a redeploy never
        # deletes datasets living under other prefixes.
        py_junk = ["**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**", "**/.venv/**"]
        code = s3deploy.BucketDeployment(self, "CodeInfraAssets", destination_bucket=bucket, destination_key_prefix="code", prune=False, retain_on_delete=False, sources=[s3deploy.Source.asset("assets")])
        for name, path in (("CodeCommon", "../common"), ("CodeSim", "../sim"), ("CodeAgent", "../agent"), ("CodeDocs", "../docs")):
            s3deploy.BucketDeployment(self, name, destination_bucket=bucket, destination_key_prefix=f"code/{path.split('/')[-1]}", prune=False, retain_on_delete=False, sources=[s3deploy.Source.asset(path, exclude=py_junk)])

        # ------------------------------------------------------------------ IAM: instance
        role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Simulated teleoperation platform instance: SSM, the exchange bucket, DCV licence, Bedrock invoke",
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
        )
        # Prefix-scoped (security review F-1): the box READS the code it boots from and WRITES only its
        # outputs. A bucket-wide grant let anything running on the instance rewrite `code/`, which user
        # data syncs and runs as root on every boot -- a persistence path for a compromised dependency.
        bucket.grant_read(role, "code/*")
        for prefix in ("datasets/*", "runs/*", "tmp/*"):
            bucket.grant_read_write(role, prefix)
        role.add_to_policy(iam.PolicyStatement(sid="DcvLicence", actions=["s3:GetObject"], resources=[f"arn:{self.partition}:s3:::dcv-license.{self.region}/*"]))
        if enable_agent_bedrock:
            # A cross-region inference profile authorises against the profile in this account AND the
            # foundation model in whichever region it routes to; both ARNs are needed.
            role.add_to_policy(
                iam.PolicyStatement(
                    sid="AgentBedrockInvoke",
                    actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                    resources=[f"arn:{self.partition}:bedrock:*::foundation-model/*", f"arn:{self.partition}:bedrock:{self.region}:{self.account}:inference-profile/*"],
                )
            )

        # ------------------------------------------------------------------ IAM: operator
        # The operator role is root on the box (ssm:SendCommand) and can read/write the bucket, so its
        # trust must name the principals (security review F-2). Trusting the account root -- every
        # principal in the account that may call sts:AssumeRole -- is allowed only as an explicit
        # opt-in for an account nobody else uses.
        if not operator_principal_arns and not allow_account_root_operator:
            raise ValueError(
                "operatorPrincipalArns is empty: pass -c operatorPrincipalArns='[\"arn:aws:iam::<account>:role/<your-role>\"]' "
                "(the principal that runs scripts/remote), or -c allowAccountRootOperator=true to trust every "
                "principal in the account (only for an account nobody else uses)."
            )
        trust = [iam.ArnPrincipal(a) for a in operator_principal_arns] if operator_principal_arns else [iam.AccountRootPrincipal()]
        operator = iam.Role(self, "OperatorRole", assumed_by=iam.CompositePrincipal(*trust), description="Drives the box over SSM and reads the bucket; no standing credentials on the box", max_session_duration=Duration.hours(12))
        bucket.grant_read_write(operator)

        # ------------------------------------------------------------------ security group
        sg = ec2.SecurityGroup(self, "InstanceSg", vpc=vpc, description="sim-teleop box: no ingress by default; add your IP for DCV 8443", allow_all_outbound=True)
        if dcv_allowed_cidr:
            for proto, note in ((ec2.Protocol.TCP, "DCV"), (ec2.Protocol.UDP, "DCV QUIC")):
                sg.add_ingress_rule(peer=ec2.Peer.ipv4(dcv_allowed_cidr), connection=ec2.Port(protocol=proto, from_port=8443, to_port=8443, string_representation="8443"), description=f"{note} from {dcv_allowed_cidr}")

        # ------------------------------------------------------------------ compute
        machine_image = ec2.MachineImage.generic_linux({self.region: ami_id}) if ami_id else ec2.MachineImage.lookup(name=ami_name_filter, owners=["amazon"])
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -xeuo pipefail",
            "exec > >(tee -a /var/log/simteleop-bootstrap.log) 2>&1",
            f"export SIMTELEOP_BUCKET={bucket.bucket_name}",
            f"export AWS_DEFAULT_REGION={self.region}",
            'echo "SIMTELEOP_BUCKET=$SIMTELEOP_BUCKET" > /etc/simteleop.env',
            f'echo "AWS_DEFAULT_REGION={self.region}" >> /etc/simteleop.env',
            f'echo "AWS_REGION={self.region}" >> /etc/simteleop.env',
            f'echo "SIMTELEOP_WORKSHOP_REPO={workshop_repo}" >> /etc/simteleop.env',
            f'echo "SIMTELEOP_WORKSHOP_COMMIT={workshop_commit}" >> /etc/simteleop.env',
            f'echo "SIMTELEOP_AGENT_MODEL={agent_model_id}" >> /etc/simteleop.env',
            "mkdir -p /opt/simteleop",
            "command -v aws >/dev/null || (apt-get update -y && apt-get install -y awscli)",
            'aws s3 sync "s3://$SIMTELEOP_BUCKET/code/" /opt/simteleop/code/ --only-show-errors',
            "chmod +x /opt/simteleop/code/bootstrap/*.sh /opt/simteleop/code/bin/* || true",
            "bash /opt/simteleop/code/bootstrap/00_bootstrap.sh",
        )
        subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC, availability_zones=[availability_zone]) if availability_zone else ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
        instance = ec2.Instance(
            self,
            "Instance",
            vpc=vpc,
            vpc_subnets=subnets,
            instance_type=ec2.InstanceType(instance_type),
            machine_image=machine_image,
            role=role,
            security_group=sg,
            user_data=user_data,
            key_name=key_name,
            associate_public_ip_address=True,
            require_imdsv2=True,
            block_devices=[ec2.BlockDevice(device_name="/dev/sda1", volume=ec2.BlockDeviceVolume.ebs(root_volume_gb, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True, delete_on_termination=True))],
        )
        instance.node.add_dependency(code)
        eip = ec2.CfnEIP(self, "Eip", domain="vpc", instance_id=instance.instance_id)

        # ------------------------------------------------------------------ operator permissions
        instance_arn = f"arn:{self.partition}:ec2:{self.region}:{self.account}:instance/{instance.instance_id}"
        operator.add_to_policy(iam.PolicyStatement(sid="SsmRunCommands", actions=["ssm:SendCommand"], resources=[instance_arn, f"arn:{self.partition}:ssm:{self.region}::document/AWS-RunShellScript"]))
        operator.add_to_policy(
            iam.PolicyStatement(
                sid="SsmStartSession",
                actions=["ssm:StartSession"],
                resources=[
                    instance_arn,
                    f"arn:{self.partition}:ssm:{self.region}::document/SSM-SessionManagerRunShell",
                    f"arn:{self.partition}:ssm:{self.region}::document/AWS-StartPortForwardingSession",
                ],
            )
        )
        operator.add_to_policy(iam.PolicyStatement(sid="SsmRead", actions=["ssm:GetCommandInvocation", "ssm:ListCommandInvocations", "ssm:ListCommands", "ssm:DescribeInstanceInformation", "ssm:DescribeSessions", "ssm:GetConnectionStatus"], resources=["*"]))
        operator.add_to_policy(iam.PolicyStatement(sid="SsmOwnSessions", actions=["ssm:TerminateSession", "ssm:ResumeSession"], resources=[f"arn:{self.partition}:ssm:{self.region}:{self.account}:session/*"]))
        operator.add_to_policy(iam.PolicyStatement(sid="Ec2Discovery", actions=["ec2:DescribeInstances", "ec2:DescribeInstanceStatus"], resources=["*"]))
        operator.add_to_policy(iam.PolicyStatement(sid="StartStopThisInstance", actions=["ec2:StartInstances", "ec2:StopInstances"], resources=[instance_arn]))

        # ------------------------------------------------------------------ outputs
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "PublicIp", value=eip.ref, description="Elastic IP (DCV endpoint)")
        CfnOutput(self, "ExchangeBucketName", value=bucket.bucket_name)
        CfnOutput(self, "OperatorRoleArn", value=operator.role_arn)
        CfnOutput(self, "DcvUrl", value=f"https://{eip.ref}:8443")
        CfnOutput(self, "SsmSessionCommand", value=f"aws ssm start-session --target {instance.instance_id} --region {self.region}")
        CfnOutput(self, "SecurityGroupId", value=sg.security_group_id)
        CfnOutput(
            self,
            "ManualStepAllowDcv",
            value=(
                f"aws ec2 authorize-security-group-ingress --group-id {sg.security_group_id} --region {self.region} --ip-permissions "
                f'"IpProtocol=tcp,FromPort=8443,ToPort=8443,IpRanges=[{{CidrIp=$(curl -s https://checkip.amazonaws.com)/32}}]" '
                f'"IpProtocol=udp,FromPort=8443,ToPort=8443,IpRanges=[{{CidrIp=$(curl -s https://checkip.amazonaws.com)/32}}]"'
            ),
            description="MANUAL: allow your current IP to reach DCV (no ingress exists by default)",
        )
        CfnOutput(self, "BootstrapLog", value="sudo tail -f /var/log/simteleop-bootstrap.log", description="First boot builds the sim image (~30-40 min); watch it here")
