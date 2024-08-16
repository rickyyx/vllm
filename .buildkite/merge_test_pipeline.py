import yaml

oss_test_path = "test-pipeline.yaml"
anyscale_test_path = "anyscale-test-pipeline.yaml"
target_merged_test_path = "merged-test-pipeline.yaml"

with open(oss_test_path, "r") as f1:
    oss_tests = yaml.safe_load(f1)

with open(anyscale_test_path, "r") as f2:
    anyscale_tests = yaml.safe_load(f2)

for i, step in enumerate(anyscale_tests["steps"]):
    step["anyscale"] = True
    anyscale_tests["steps"][i] = step

merged_steps = {"steps": oss_tests["steps"] + anyscale_tests["steps"]}
with open(target_merged_test_path, 'w') as outfile:
    yaml.dump(merged_steps, outfile, default_flow_style=False)
