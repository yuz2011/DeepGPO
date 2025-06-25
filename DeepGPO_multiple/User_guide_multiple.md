# User Guide for Doubly Glycosylated Peptides (`DeepGPO_multiple`)

This document describes how to process and predict MS/MS spectra of **doubly glycosylated peptides** using the `DeepGPO_multiple` module. Compared to the main `DeepGPO`, this version incorporates minor modifications in data processing to support Byonic outputs and prediction of fragment ion intensities for doubly glycosylated glycopeptides.

---
## Example data
The example data can be found in the `multiple_data/demo` folder. 
Spectra files:
`demo_mgf/2019_09_16_StcEmix_35trig_EThcD15_rep1_HCDFT.mgf`

In order to obtain an MGF file, you will need to employ a program or tool with the ability to convert your data into MGF format. Here, the MGF files are from the searching process of pGlyco3 (release: pGlyco3.1, https://github.com/pFindStudio/pGlyco3/releases). When multiple MGF files are stored under a specified folder, they can be automatically retrieved in one go.

Sequence searching software results:
`StcEmix_35trig_EThcD15_rep1_GlycoPSMs.txt`

The `StcEmix_35trig_EThcD15_rep1_GlycoPSMs.txt` file contains the Byonic results for glycopeptides. Here, this file is directly downloaded from the original literature. Furthermore, if you have search results from other tools, they can be utilized by converting their format to match that of pGlyco3.

Model Used:
https://drive.google.com/drive/folders/1PT_1bVlbjwSgmaq5i9wsNC0PHwdDcw0y?usp=drive_link

The model files include both the base model and the trained model. They can be downloaded from the link above. It is recommended to place them in the multiple_data folder for easier access.

## Example Workflow
### 1.Entry to the folder including DeepGPO_multiple code files.
Users can navigate to the relevant folder using a command such as cd D:\code\DeepGPO_multiple\multiple_code. The path “D:\code\DeepGPO_multiple\multiple_code” signifies the directory containing the Python scripts for DeepGPO_multiple.

### 2. Data Preprocessing

Convert the Byonic identification results and experimental spectra into a standardized `.csv` format.

```
python 1_dataset_format_NO_multiple.py --datafold ../multiple_data/demo/ --DFNAME StcEmix_35trig_EThcD15_rep1_GlycoPSMs.txt --mgfdatafold demo_mgf --mgfsourceorign pGlyco3 --only_duplicated Retained_all --output_name Demo_doubly_data_1st.csv
```

### 3. Model Training

Train a DeepGPO model using your processed doubly glycosylated peptide dataset.

```
python 2_train_byBY_multiple.py --model_ablation All_base --testdata alltest --task_name multiples --folder_path ../multiple_data/ --organism demo --trainpathcsv train_ETD_EThCD_combine.csv --pattern *_data_1st.csv
```

### 4. Prediction

Use the trained model to predict MS/MS spectra from new doubly glycosylated peptide data.

```
python 3_replace_predict_byBY_multiple.py --datafold ../multiple_data/ --trainpathcsv ../multiple_data/demo/Demo_doubly_data_1st.csv --bestmodelpath ../multiple_data/model/train_model/epoch-64_step-448_mediancos-0.928872.pt --savename test_byBY_multiple --ms2_method cos --postprocessing off
```
Here,the trained model is also provided,you can also replace `<model_path>.pt` with the actual path to your trained model checkpoint.
For full parameter descriptions, refer to the main [`user_guide.md`](../User_guide.md).