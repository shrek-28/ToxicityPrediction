import joblib
import xgboost as xgb
import argparse
import os
import sys

def convert_pkl_to_json(pkl_path, output_path=None):
    try:
        model = joblib.load(pkl_path)
    except Exception as e:
        print(f"❌ Failed to load .pkl model: {e}")
        sys.exit(1)

    # Check for XGBClassifier
    if not isinstance(model, xgb.XGBClassifier):
        print("❌ The loaded model is not an XGBClassifier.")
        sys.exit(1)

    # Save the booster part as a .json
    if output_path is None:
        output_path = os.path.splitext(pkl_path)[0] + ".json"

    try:
        booster = model.get_booster()
        booster.save_model(output_path)
        print(f"✅ Model saved in .json format at: {output_path}")
    except Exception as e:
        print(f"❌ Could not convert to .json: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XGBoost .pkl model to .json format")
    parser.add_argument("pkl_path", help="Path to .pkl model file")
    parser.add_argument("--output", help="Optional path for .json output")

    args = parser.parse_args()
    convert_pkl_to_json(args.pkl_path, args.output)
