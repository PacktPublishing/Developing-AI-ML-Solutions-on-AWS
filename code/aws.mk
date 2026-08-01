# Shared across chapters. Chapter Makefiles export dummy credentials
# (local/localsecret) so the local stack never sees anything real; targets
# that talk to real AWS must shed them again or boto3 will prefer the dummy
# env keys over the profile chain. Run those commands through $(REAL_AWS):
#
#   include ../aws.mk
#   some-cloud-target:
#   	$(REAL_AWS) uv run script.py
REAL_AWS := env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY
