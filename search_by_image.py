import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--diretory",
    required=True,
    help="Image path")

print("Searching images...")
