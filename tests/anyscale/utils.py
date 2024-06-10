# This library may only be used in the Anyscale Platform.
# Notwithstanding the terms of any license or notice within this container,
# you may not modify, copy or remove this file.
# Your right to use this library is subject to the
# Anyscale Terms of Service (anyscale.com/terms)
# or other written agreement between you and Anyscale.

# Copyright (2023 and onwards) Anyscale, Inc.
# This Software includes software developed at Anyscale (anyscale.com/)
# and its use is subject to the included LICENSE file.

import json
import logging
import os

import boto3

ENV_TOKEN_OVERRIDES = os.getenv("AVIARY_ENV_AWS_SECRET_NAME",
                                "aviary/env_overrides")

logger = logging.getLogger(__name__)


# Copied from aviary
class SecretManager:

    def __init__(self, secret_name: str = ENV_TOKEN_OVERRIDES):
        self.secret_overrides = self.get_all_secrets(secret_name)

    def get_all_secrets(self, secret_name: str):
        try:
            aws_region_name = os.getenv("AWS_REGION", "us-west-2")

            # Create a Secrets Manager client
            session = boto3.session.Session()
            client = session.client(service_name="secretsmanager",
                                    region_name=aws_region_name)
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name)

            # Decrypts secret using the associated KMS key.
            secret = get_secret_value_response["SecretString"]

            secret_dict = json.loads(secret)
            return secret_dict
        except Exception as e:
            print(f"Unable to load env override secrets from {secret_name}. "
                  f"Using default secrets from env. {e}")
            return {}

    def override_secret(self, env_var_name: str, set_in_env=True):
        secret = self.get_secret_value(env_var_name, default_value=None)
        if secret is None:
            print(f"Secret {env_var_name} was not found.")
        elif set_in_env:
            os.environ[env_var_name] = secret
            print(f"Secret {env_var_name} was set in the env.")
        return secret

    def get_secret_value(self, secret_name: str, default_value=None) -> str:
        # First read from env var, then from aws secrets
        secret = os.getenv(
            secret_name, self.secret_overrides.get(secret_name, default_value))
        return secret
