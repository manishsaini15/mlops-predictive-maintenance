import yaml
import subprocess

with open("pipeline.yml", "r") as f:
    pipeline = yaml.safe_load(f)

for step in pipeline["steps"]:

    print(f"\n Running: {step['name']}")

    result = subprocess.run(
        ["python", step["script"]],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"Step failed: {step['name']}")

print("\n Pipeline completed successfully!")