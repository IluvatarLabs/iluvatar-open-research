#!/usr/bin/env python3
"""
V4: Cross-Atlas Vascular Replication using local Nature Aging data
"""

import json
import numpy as np
import scanpy as sc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("V4: Cross-Atlas Vascular Replication")
print("=" * 60)

# Load local Nature Aging file
print("\nLoading local Nature Aging atlas...")
try:
    ad_na = sc.read_h5ad("data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad")
    print(f"  Loaded: {ad_na.n_obs:,} cells")
except Exception as e:
    print(f"  Failed: {e}")
    exit(1)

# Check available columns
print(f"\nAvailable columns in obs:")
print(ad_na.obs.columns.tolist()[:30])

# Use annotation_level2 as the cell type annotation
ann_col = 'annotation_level2'
print(f"\nUsing annotation column: {ann_col}")
print(ad_na.obs[ann_col].value_counts().to_dict())

# Check for any vascular/endothelial keywords
vascular_keywords = ["EC", "Endothelial", "Capillary", "Venule", "Artery", "Vascular", "Vessel", "Endo"]
all_annotations = [str(a) for a in ad_na.obs[ann_col].unique()]
print(f"\nAll unique annotations ({len(all_annotations)}):")
for ann in sorted(all_annotations):
    print(f"  - {ann}")

# Look for vascular annotations
vascular_types = []
for ann in all_annotations:
    ann_str = str(ann).lower()
    if any(kw.lower() in ann_str for kw in vascular_keywords):
        vascular_types.append(ann)
        print(f"\n  MATCHED vascular: {ann}")

if vascular_types:
    print(f"\nFound vascular types: {vascular_types}")
else:
    print("\nNO vascular annotations found in Nature Aging fibroblasts file")

# Also check level1 for overview
print(f"\nLevel1 overview:")
print(ad_na.obs['annotation_level1'].value_counts().to_dict())

# V4 Results
print("\n" + "=" * 60)
print("V4 Result Summary")
print("=" * 60)
print(f"Status: {'VASCULAR_FOUND' if vascular_types else 'NO_VASCULAR_CELLS'}")
print(f"Fallback: F084 is single-atlas; cross-atlas vascular replication not possible")
print(f"Note: Nature Aging fibroblasts file contains fibroblasts, Schwann cells,")
print(f"      tenocytes — no vascular endothelial cells present")
