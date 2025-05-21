# %%
import pandas as pd

# %%
df_sv = pd.read_csv("llava_sv_puzzles.csv")
df_cot = pd.read_csv("llava_cot_puzzles_2.csv")
# %%
df_cot.columns
# %%
df_sv.columns
# %%
cot_accuracy = sum(df_cot["cot_correct"]) / len(df_cot["cot_correct"])
direct_accuracy = sum(df_cot["direct_correct"]) / len(df_cot["direct_correct"])
print(f"CoT accuracy: {cot_accuracy} - Direct accuracy: {direct_accuracy}")
# %%
sv_accuracy = sum(df_sv["verified_correct"]) / len(df_sv["verified_correct"])
initial_accuracy = sum(df_sv["initial_correct"]) / len(df_sv["initial_correct"])
direct_accuracy = sum(df_sv["direct_correct"]) / len(df_sv["direct_correct"])
print(
    f"SV accuracy: {sv_accuracy} - Initial accuracy: {initial_accuracy} - Direct accuracy: {direct_accuracy}"
)
# %%
