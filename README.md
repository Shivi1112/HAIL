# HAIL

Install Requirements:

`pip install -r tasks/requirements.txt`

Create train/val/test for e.g. **Mortality Prediction**:

```
python tasks/mp/mp.py \
 --mimic_dir {MIMIC_DIR} \   # required
 --save_dir {DIR_TO_SAVE_DATA} \   # required
 --admission_only True \   # required
```

_mimic_dir_: Directory that contains unpacked NOTEEVENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES_ICD.csv

_save_dir_: Any directory to save the data

_admission_only_: True=Create simulated Admission Notes, False=Keep complete Discharge Summaries

run convert_para_to_csv.py

then run model.py

_Apply these scripts accordingly for the other outcome tasks:

**Length-of-Stay** (los/los.py), 
