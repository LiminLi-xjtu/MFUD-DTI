# MFUD-DTI

## Overview
MFUD-DTI: Drug-Target Interaction Prediction via Molecular Functional Unit Docking
<img src="MFDU.png" width="600"/>

## 1. Environment Setup

It is recommended to use `conda` to create an isolated Python environment.

```bash
# Create and activate the environment
conda create --name mfud_env python=3.8
conda activate mfud_env

# Install PyTorch and related Graph Neural Network libraries (Adjust based on your CUDA version)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install torch_cluster-1.6.3+pt20cu117-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt20cu117-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.18+pt20cu117-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt20cu117-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==2.6.1
pip install rdkit-pypi==2022.09.5

# Install other necessary dependencies
pip install scipy biopython pandas biopandas timeout_decorator py3Dmol umap-learn plotly mplcursors lifelines reprint tqdm
pip install "fair-esm"
```

## 2. Data Processing and Structure

Data is stored in the `dataset/` directory with the following structure:

- **CSV Files**: Stored in `dataset/[Dataset Name]/[Split Type]/` (e.g., `dataset/Davis/scaffold/`).
  - Must contain `Protein` (amino acid sequence) and `Ligand` (SMILES) columns.
  - Must contain a label column (e.g., `classification_label`).
- **Preprocessed Graph Data**: 
  - Upon the first run, the system automatically converts sequences and SMILES into graph data, saving them as `protein.pt` and `ligand.pt`.
  - Subsequent runs will load these `.pt` files directly to accelerate startup.

## 3. Running the Code

### 3.1 Running a Single Experiment
Use `main.py` for training and evaluation:

```bash
python main.py 
    --config_path ./config/davis/config_davis_sp.json 
    --datafolder ./dataset/Davis/scaffold_protein 
    --datafolder_pt ./dataset/Davis 
    --result_path ./test_result/Davis/scaffold_protein_test 
    --epochs 20 
    --batch_size 64
```

### 3.2 Batch Automated Execution
Batch execution scripts for different datasets are provided. These scripts automatically iterate through different split types (random, scaffold, etc.) and repeat each run 5 times to obtain average metrics.

```bash
# Run Davis dataset experiments
bash davis_run.sh

# Run BioSNAP dataset experiments
bash biosnap_run.sh

# Run DrugBank dataset experiments
bash drugbank_run.sh
```

## 4. Result Summary and Metric Calculation

After training, results are saved in the `result/` or `test_result/` directory. You can use `results.ipynb` or `summary_metrics_from_csv.py` within each dataset directory to aggregate mean and standard deviation across 5 runs and generate comparison charts:

```bash
python result/Davis/summary_metrics_from_csv.py
```

## 5. Directory Description

- `config/`: JSON configuration files for various experimental settings across datasets.
- `models/`: Core model architectures, including physicochemical feature extraction and attention mechanisms.
- `utils/`: Data loaders, training engines, and metric evaluation tools.
- `screening.py`: Used for large-scale virtual screening on new data.