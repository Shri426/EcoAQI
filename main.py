import subprocess

print("Running preprocessing...")
subprocess.run(["python", "src/preprocessing.py"])

print("Training model...")
subprocess.run(["python", "src/model.py"])

print("Explaining AQI...")
subprocess.run(["python", "src/explain.py"])
