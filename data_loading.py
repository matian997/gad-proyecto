import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    required=True,
    help="Path to the directory of images")

print("Importing images...")
