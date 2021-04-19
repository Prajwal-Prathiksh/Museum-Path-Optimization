# Imports
import __init__
import os
import numpy as np
from code.data_input.input_final import get_input_loader

loader = get_input_loader("Choose_TC_Asym_NPZ.txt", False)
print("Solving assymmetric problem...")

tc_number = 5

tc_name = loader.get_test_case_name(tc_number)
cost_matrix = loader.get_input_test_case(tc_number).get_cost_matrix()

print(tc_name, "cost matrix:")
print(cost_matrix)

OUTPUT_DIR = os.path.join(os.getcwd(), "output", "for_mridul")
if os.path.exists(OUTPUT_DIR) is False:
    os.mkdir(OUTPUT_DIR)

fname = os.path.join(OUTPUT_DIR, f"{tc_name}_data.npz")
np.savez(fname, tc_name=tc_name, cost_matrix=cost_matrix)
print(f"\nSummary data saved at: {fname}")
