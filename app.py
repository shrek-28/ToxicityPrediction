import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, GraphDescriptors
from rdkit import RDLogger
import sys

# === Suppress RDKit warnings ===
RDLogger.DisableLog('rdApp.*')

# === Config ===
MODEL_PATHS = {
    "NR-AhR": "Models/NR-AhR/lr_rf_xgb_stack.pkl",
    "NR-AR": "Models/NR-AR/xgboost_smoteen.pkl",
    "NR-AR-LBD": "Models/NR-AR-LBD/xgboost_smoteen_rs_spw1.pkl",
    "NR-Aromatase": "Models/NR-Aromatase/Voting_Classifiers/lr_rf_xgboost_knn_et_svc_gb.pkl", 
    "NR-ER": "Models/NR-ER/Voting_classifiers/lr_rf_xgb.pkl", 
    "NR-ER-LBD": "Models/NR-ER-LBD/voter_lr_et.pkl", 
    "NR-PPAR-gamma": "Models/NR-PPAR-gamma/xgboost_smoteen_spwformula.pkl", 
    "SR-ARE": "Models/SR-ARE/smoteen_enggfeatures_xgboost.pkl", 
    "SR-ATAD5": "Models/SR-ATAD5/V_classifiers/rf_xgb_voter.pkl", 
    "SR-HSE": "Models/SR-HSE/xgboost_smoteen.pkl", 
    "SR-MMP": "Models/SR-MMP/xgboost_smoteen.pkl", 
    "SR-p53": "Models/SR-p53/rsampling_smoteen_ovr.pkl"
}

BASE_FEATURES = [
    'MolecularWeight', 'LogP', 'TPSA', 'HBDonors', 'HBAcceptors',
    'RotatableBonds', 'FractionCSP3', 'HeavyAtoms', 'RingCount', 'LogS_ESOL',
    'FormalCharge', 'AromaticRings', 'AromaticHeterocycles',
    'AliphaticRings', 'MolecularComplexity', 'MolarRefractivity'
]

ENGINEERED_FEATURES = ['TPSA_LogP', 'MW_per_HBD', 'LogP_div_HBA']
FULL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# === Feature Computation ===
def compute_all_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)

        # Base descriptors
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        ring_count = mol.GetRingInfo().NumRings()
        logS_esol = -0.74 * logp + 0.003 * mw - 0.49 * tpsa - 0.003 * rot_bonds + 0.16
        formal_charge = Chem.GetFormalCharge(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        aromatic_heterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
        aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
        complexity = GraphDescriptors.Chi0(mol)
        molar_refractivity = Crippen.MolMR(mol)

        # Engineered features
        tpsa_logp = tpsa * logp
        mw_per_hbd = mw / (hbd + 1e-6)
        logp_div_hba = logp / (hba + 1e-6)

        return [
            mw, logp, tpsa, hbd, hba,
            rot_bonds, fraction_csp3, heavy_atoms, ring_count, logS_esol,
            formal_charge, aromatic_rings, aromatic_heterocycles,
            aliphatic_rings, complexity, molar_refractivity,
            tpsa_logp, mw_per_hbd, logp_div_hba
        ]
    except:
        return None

# === Load all models ===
models = {}
for target, path in MODEL_PATHS.items():
    try:
        with open(path, 'rb') as f:
            models[target] = pickle.load(f)
        print(f"✅ Loaded model for {target}")
    except Exception as e:
        print(f"❌ Failed to load model for {target}: {e}")
        sys.exit(1)

# # === Load SMILES file ===
# input_csv = "input_smiles.csv"       # <-- Replace or make dynamic
# output_csv = "toxicity_predictions.csv"

# df = pd.read_csv(input_csv)
# if "SMILES" not in df.columns:
#     print("❌ Input CSV must have a column named 'SMILES'")
#     sys.exit(1)

# # === Compute features ===
# features_list = []
# valid_indices = []

# for idx, smiles in df["SMILES"].items():
#     feats = compute_all_features(smiles)
#     if feats is not None:
#         features_list.append(feats)
#         valid_indices.append(idx)
#     else:
#         print(f"⚠️ Skipping invalid SMILES at index {idx}: {smiles}")

# # === Prepare feature DataFrame ===
# X_full = pd.DataFrame(features_list, columns=FULL_FEATURES)
# df_valid = df.loc[valid_indices].reset_index(drop=True)

# # === Run predictions ===
# for target, model in models.items():
#     try:
#         if target == "SR-ARE":
#             X_input = X_full[FULL_FEATURES]  # all features
#         else:
#             X_input = X_full[BASE_FEATURES]  # only base

#         probs = model.predict_proba(X_input)[:, 1]
#         preds = (probs >= 0.5).astype(int)

#         df_valid[f"{target}_Prediction"] = preds
#         df_valid[f"{target}_Probability"] = probs
#     except Exception as e:
#         print(f"❌ Prediction for {target} failed: {e}")

# # === Save output ===
# df_valid.to_csv(output_csv, index=False)
# print(f"\n✅ Predictions saved to: {output_csv}")

if __name__ == "__main__":
    import streamlit as st
    import tempfile
    import os

    st.set_page_config(page_title="Toxicity Predictor", layout="centered")
    st.title("☣️ Molecular Toxicity Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'SMILES' column", type=["csv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.read())
            input_csv = tmp.name

        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            st.error(f"❌ Error reading uploaded file: {e}")
            os.unlink(input_csv)
            st.stop()

        if "SMILES" not in df.columns:
            st.error("❌ Uploaded CSV must contain a column named 'SMILES'")
            os.unlink(input_csv)
            st.stop()

        st.success("✅ File uploaded successfully. Running predictions...")

        # === Compute features ===
        features_list = []
        valid_indices = []

        for idx, smiles in df["SMILES"].items():
            feats = compute_all_features(smiles)
            if feats is not None:
                features_list.append(feats)
                valid_indices.append(idx)
            else:
                st.warning(f"⚠️ Skipping invalid SMILES at index {idx}: {smiles}")

        if not features_list:
            st.error("❌ No valid SMILES found.")
            os.unlink(input_csv)
            st.stop()

        X_full = pd.DataFrame(features_list, columns=FULL_FEATURES)
        df_valid = df.loc[valid_indices].reset_index(drop=True)

        # === Run predictions ===
        for target, model in models.items():
            try:
                if target == "SR-ARE":
                    X_input = X_full[FULL_FEATURES]
                else:
                    X_input = X_full[BASE_FEATURES]

                probs = model.predict_proba(X_input)[:, 1]
                preds = (probs >= 0.5).astype(int)

                df_valid[f"{target}_Prediction"] = preds
                df_valid[f"{target}_Probability"] = probs
            except Exception as e:
                st.warning(f"❌ Prediction for {target} failed: {e}")

        # === Output ===
        st.success("✅ Predictions complete!")
        st.dataframe(df_valid)

        # === Download ===
        output_csv = "toxicity_predictions.csv"
        df_valid.to_csv(output_csv, index=False)

        with open(output_csv, "rb") as f:
            st.download_button(
                label="📥 Download Predictions CSV",
                data=f,
                file_name="toxicity_predictions.csv",
                mime="text/csv"
            )

        # Cleanup temp file
        os.unlink(input_csv)
