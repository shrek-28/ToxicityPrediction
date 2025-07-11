# Data Cleaning, Preparation and EDA

## Initial Data Exploration
* Loaded the Tox21 dataset and performed preliminary inspection using ```.head()```, ```.info()```, and ```.describe()``` to understand the structure and summary statistics.
* Identified categorical (e.g., target labels) and continuous variables (e.g., molecular descriptors), along with their frequency distributions via .value_counts().
* Conducted a null value analysis, calculating both the total missing values and the percentage of missing data per column, to inform downstream imputation and data cleaning strategies.

### Label Distribution 
| Target        | Non-Toxic (0) | Toxic (1) | Total |
| ------------- | ------------- | --------- | ----- |
| NR-AR         | 6956          | 309       | 7265  |
| NR-AR-LBD     | 6521          | 237       | 6758  |
| NR-AhR        | 5781          | 768       | 6549  |
| NR-Aromatase  | 5521          | 300       | 5821  |
| NR-ER         | 5400          | 793       | 6193  |
| NR-ER-LBD     | 6605          | 350       | 6955  |
| NR-PPAR-gamma | 6264          | 186       | 6450  |
| SR-ARE        | 4890          | 942       | 5832  |
| SR-ATAD5      | 6808          | 264       | 7072  |
| SR-HSE        | 6095          | 372       | 6467  |
| SR-MMP        | 4892          | 918       | 5810  |
| SR-p53        | 6351          | 423       | 6774  |

> All targets are skewed toward the negative class, highlighting the importance of class balancing techniques (e.g., SMOTE, class weights) during model training.

### Missing Value Analysis 
| Column        | % Missing Values |
| ------------- | ---------------- |
| NR-AR         | 7.23%            |
| NR-AR-LBD     | 13.70%           |
| NR-AhR        | 16.37%           |
| NR-Aromatase  | 25.67%           |
| NR-ER         | 20.92%           |
| NR-ER-LBD     | 11.19%           |
| NR-PPAR-gamma | 17.64%           |
| SR-ARE        | 25.53%           |
| SR-ATAD5      | 9.69%            |
| SR-HSE        | 17.42%           |
| SR-MMP        | 25.81%           |
| SR-p53        | 13.50%           |
| mol\_id       | 0.00%            |
| smiles        | 0.00%            |

> 96 rows were identified to have missing values across 11 out of the 12 targets, indicating incomplete toxicity profiling. These rows were flagged for either removal or special treatment depending on downstream modeling requirements.

## Feature Engineering (Molecular Descriptor Generation)
To represent each compound numerically for machine learning, a set of physicochemical and structural descriptors were computed from SMILES using RDKit and other cheminformatics tools. These features capture the molecule’s bioactivity-relevant properties and were used as input to the classification models.

### Atomic and Physicochemical Properties:
| Descriptor            | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `total_atoms`         | Total number of atoms in the molecule                       |
| `aromatic_atoms`      | Count of atoms in aromatic systems                          |
| `aromatic_proportion` | Ratio of aromatic atoms to total atoms                      |
| `logP`                | Octanol-water partition coefficient; measures lipophilicity |
| `molwt`               | Molecular weight                                            |
| `rot_bonds`           | Number of rotatable bonds (flexibility proxy)               |
| `logS`                | Logarithm of aqueous solubility                             |
| `positive_charge`     | Total formal positive charge                                |
| `negative_charge`     | Total formal negative charge                                |

### Ring Systems and Structural Complexity
| Descriptor              | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `aromatic_rings`        | Number of aromatic ring systems                       |
| `aromatic_heterocycles` | Aromatic rings containing heteroatoms                 |
| `aliphatic_rings`       | Saturated ring count                                  |
| `molecular_complexity`  | Estimate of topological and stereochemical complexity |
| `molar_refractivity`    | Proxy for polarizability and van der Waals volume     |

### Functional Groups and Substructures
| Descriptor        | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `heteroatoms`     | Count of non-carbon atoms (e.g., N, O, S)                 |
| `halogencount`    | Number of halogen atoms (F, Cl, Br, I)                    |
| `phenolic_groups` | Count of phenol-like OH groups attached to aromatic rings |

> These descriptors were selected for their relevance to ADME-Tox behavior, bioavailability, and receptor binding — all crucial factors in toxicity prediction.

## Filling In Missing Data
To address the non-random pattern of missing values in the Tox21 dataset — particularly among toxicity labels — two imputation methods were systematically evaluated:
* K-Nearest Neighbors (KNN) Imputation
* Multivariate Imputation by Chained Equations (MICE)

The goal was to impute missing toxicity labels in a manner that preserves both statistical 
structure and biological interpretability.

Visit these links for a further idea on [KNN Imputation](https://www.geeksforgeeks.org/machine-learning/handling-missing-data-with-knn-imputer/) and [MICE](https://www.machinelearningplus.com/machine-learning/mice-imputation/)

### KNN Imputation
KNN imputation fills missing values by averaging the corresponding values of the k most similar samples, based on available features (molecular descriptors). Various combinations of ```k``` and weighting strategies were tested and evaluated using accuracy and F1-score on a masked subset (simulating missingness).

| k  | Weights  | Accuracy | F1-score   |
| -- | -------- | -------- | ---------- |
| 3  | uniform  | 0.9198   | 0.2350     |
| 3  | distance | 0.9174   | 0.2829     |
| 5  | uniform  | 0.9247   | 0.1626     |
| 5  | distance | 0.9264   | **0.2878** |
| 7  | uniform  | 0.9222   | 0.1833     |
| 7  | distance | 0.9320   | 0.2639     |
| 10 | uniform  | 0.9255   | 0.1567     |
| 10 | distance | 0.9246   | 0.2075     |

* Best configuration: ```k=5```, ```weights='distance'```
* Best F1-score: 0.2878

KNN is biologically meaningful in this context because:
* Structurally similar compounds often behave similarly in biological systems (QSAR principle).
* Molecular descriptors capture these structural and physicochemical properties.
* Using distance-weighted neighbors helps prioritize closer (and thus more bio-relevant) compounds during imputation.

### MICE (Multivariate Imputation by Chained Equations)
MICE models each variable with missing data as a function of the other variables iteratively, thereby preserving multivariate relationships across the dataset. A simulation with 10% randomly masked values yielded:
* **Accuracy**: 0.9400
* **Weighted F1-score**: 0.9309

This significant improvement in both performance metrics over KNN highlights the strength of MICE in this context.

Molecular descriptors are not independent — for example, molecular weight often correlates with logP and rotatable bonds. MICE captures these complex interdependencies, producing more biologically plausible imputations. Reflects how biological properties are co-regulated and interdependent (e.g., size, charge, polarity affecting permeability or toxicity simultaneously).

Based on both quantitative performance and biological interpretability, MICE was selected as the preferred imputation method. It more accurately reconstructs the multivariate relationships that underpin molecular toxicity, especially in high-dimensional descriptor space.

## Exploratory Data Analysis
Initial exploratory data analysis was conducted to understand the distribution, variance, and interdependencies of molecular descriptors used in toxicity prediction. This step was critical for uncovering patterns, identifying outliers, and informing both feature selection and downstream imputation strategies.

### Boxplots:
Boxplots were used to examine the spread and outliers in continuous molecular descriptors such as ```logP```, ```TPSA```, ```molecular_weight```, and ```rotatable_bonds```.

* Boxplots help identify extreme values which may reflect either chemically atypical molecules (e.g., highly polar or large synthetic compounds) or data entry errors.
* Understanding the variability in features like ```logP``` and ```molwt``` is critical, as these directly influence **absorption**, **distribution**, and **membrane permeability**, which are upstream determinants of toxicity.

### Histograms
Histograms were plotted to assess the univariate distribution of key descriptors across the dataset.

* Histograms reveal skewness, modality, and sparsity — essential for deciding on normalization or transformation techniques.
* For example, a heavily right-skewed distribution in ```rot_bonds``` or ```molwt``` might indicate a prevalence of small druglike compounds, with occasional large outliers.
* Helps verify that the dataset reflects a realistic chemical diversity relevant to pharmacological toxicity.

### Scatterplots
Pairwise scatterplots (or scatter matrices) were generated to explore **bivariate relationships** among descriptors.

* Scatterplots can uncover correlated features, such as between ```molecular_weight``` and ```TPSA```, or between ```logP``` and ```HBAcceptors```.
* Detecting **descriptor collinearity** is essential for preventing redundancy in the model and avoiding overfitting.
* Biologically, these correlations reflect **physicochemical interdependence** — such as how increasing polarity (TPSA) often correlates with decreased lipophilicity (logP), influencing compound behavior in vivo.



