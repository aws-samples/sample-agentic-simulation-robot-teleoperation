#!/usr/bin/env python3
"""CDK app for the simulated teleoperation platform: one GPU EC2 box, an S3 exchange bucket, roles.

Context-driven, nothing to edit:

    cdk deploy -c dcvAllowedCidr=$(curl -s https://checkip.amazonaws.com)/32
    cdk deploy -c instanceType=g6e.4xlarge -c availabilityZone=eu-central-1b
"""
from __future__ import annotations

import os
import sys

import aws_cdk as cdk

from infra.simteleop_stack import SimTeleopPlatformStack

# The solution identity lives in common/simteleop_common/solution.json; the same reader runs on the
# instance (installed package) and here (checkout). Raises at synth if the block is missing, so a stack
# can never deploy with a description that is not counted.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from simteleop_common.solution import stack_description  # noqa: E402

app = cdk.App()


def ctx(key: str, default=None):
    v = app.node.try_get_context(key)
    return default if v is None or v == "" else v


def _list(v):
    """Context values arrive as a JSON list from cdk.json or as a string from -c (JSON or comma-separated)."""
    if isinstance(v, (list, tuple)):
        return list(v)
    v = str(v).strip()
    if v.startswith("["):
        import json
        return json.loads(v)
    return [x.strip() for x in v.split(",") if x.strip()]


account = ctx("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = ctx("region") or os.environ.get("CDK_DEFAULT_REGION")

SimTeleopPlatformStack(
    app,
    ctx("stackName", "SimTeleopPlatform"),
    env=cdk.Environment(account=account, region=region),
    description=stack_description(),
    instance_type=ctx("instanceType", "g7e.4xlarge"),
    ami_id=ctx("amiId"),
    ami_name_filter=ctx("amiNameFilter", "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260623"),  # last 580-driver build; see the stack
    root_volume_gb=int(ctx("rootVolumeGb", 400)),
    availability_zone=ctx("availabilityZone"),
    max_azs=int(ctx("maxAzs", 3)),
    dcv_allowed_cidr=ctx("dcvAllowedCidr"),
    operator_principal_arns=_list(ctx("operatorPrincipalArns", [])),
    allow_account_root_operator=str(ctx("allowAccountRootOperator", "false")).lower() == "true",
    key_name=ctx("keyName"),
    workshop_repo=ctx("workshopRepo", "https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git"),
    workshop_commit=ctx("workshopCommit", "ce807d99724cb65671abec01f908a2fcb4a6eab7"),
    agent_model_id=ctx("agentModelId", "global.openai.gpt-6-astra"),
    enable_agent_bedrock=str(ctx("enableAgentBedrock", "true")).lower() == "true",
    retain_data_on_destroy=str(ctx("retainDataOnDestroy", "true")).lower() == "true",
)

app.synth()
