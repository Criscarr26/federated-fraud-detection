"""Small script to validate that required packages and project modules import correctly.

Run with: python tools/validate_imports.py
"""
import importlib
import sys

packages = [
    "torch",
    "flwr",
    "sklearn",
    "pandas",
    "matplotlib",
    "joblib",
    "numpy",
]

modules = [
    "fed_fraud.data_utils",
    "fed_fraud.models",
    "fed_fraud.privacy",
    "fed_fraud.client_pytorch",
    "fed_fraud.server_pytorch",
    "fed_fraud.experiments",
]

def check_imports():
    ok = True
    print("Checking external packages:")
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"  OK: {pkg}")
        except Exception as e:
            print(f"  FAIL: {pkg} -> {e}")
            ok = False

    print("\nChecking project modules:")
    for m in modules:
        try:
            importlib.import_module(m)
            print(f"  OK: {m}")
        except Exception as e:
            print(f"  FAIL: {m} -> {e}")
            ok = False

    if not ok:
        print("\nSome imports failed. Please install missing packages or fix path issues.")
        sys.exit(1)
    print("\nAll imports OK")


if __name__ == "__main__":
    check_imports()
