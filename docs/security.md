# Security

How the deployment is protected, what the trade-offs of a single-user research instance are, and
what to change before sharing the instance or the account.

## In place

- No inbound rule; shell access over SSM only; DCV port 8443 opens only for the one CIDR you pass.
- The instance role reads `code/` in its bucket and writes only `datasets/`, `runs/` and `tmp/`, so
  the code the instance executes at boot cannot be rewritten from the instance. The bucket is
  versioned, encrypted, TLS-only and blocks public access. The role also reads the regional DCV
  licence bucket and may call `bedrock:InvokeModel*`.
- The operator role (root on the instance via SSM, start/stop, the bucket) trusts only the principals
  named in `operatorPrincipalArns`; it cannot terminate the instance, see other buckets or call
  Bedrock.
- IMDSv2 required, EBS encrypted, no long-lived credentials anywhere: the agent uses the instance
  role, and `/etc/simteleop.env` holds only identifiers.
- The model cannot change what counts as success or reach a shell: tools take typed arguments, the
  environment server clamps them to joint limits, verdicts are computed from simulator state, and the
  episode runner re-judges every episode independently of the agent's claim.

## Known trade-offs of a single-user research instance

Change these before sharing the instance or the account:

- The simulator container runs with `--gpus all --network host` (Isaac Sim needs the GPU; the
  upstream workshop runs this way). The environment server is unauthenticated on `127.0.0.1:7010`:
  any local process on the instance can drive the robot or stop a recording.
- DCV authenticates with the `ubuntu` Linux password on a public IP (from your CIDR only). Prefer
  `aws ssm start-session --document-name AWS-StartPortForwardingSession` and no ingress at all.
- All egress is open (needed at first boot), and there are no flow logs, S3 access logs or alarms.
- Bedrock is allowed on any foundation model in any region (a global inference profile routes
  anywhere); scope it to your profile if spend matters.
- Some dependencies are open-range (`numpy`, `Pillow`, `boto3`) and the NGC base image is pulled by
  tag; the DCV tarball's signature is not verified at install.

## Your part

Keep `dcvAllowedCidr` to your own address and set a strong `ubuntu` password if you use the desktop;
name only the principals that should be root on the instance; publish datasets you care about
(`simteleop-publish`) — the instance has no termination protection.

Not covered: the contents of the simulator image (about 29 GB; no CVE scan is part of the build) and
the upstream workshop code, which is used as pinned.
