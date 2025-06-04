# %%
import pandas as pd

# %%
df_sv = pd.read_csv("../results/sv_results.csv")
df_cot = pd.read_csv("../results/cot_results.csv")
# %%
df_cot.columns
# %%
df_sv.columns
# %%
cot_accuracy = sum(df_cot["cot_correct"]) / len(df_cot["cot_correct"])
direct_accuracy = sum(df_cot["direct_correct"]) / len(df_cot["direct_correct"])
print(f"CoT accuracy: {cot_accuracy} - Direct accuracy: {direct_accuracy}")
# %%
sv_accuracy = sum(df_sv["answer"] == df_sv["final_answer"]) / len(df_sv["answer"])
initial_accuracy = sum(df_sv["answer"] == df_sv["initial_answer"]) / len(df_sv["answer"])
direct_accuracy = sum(df_sv["answer"] == df_sv["direct_answer"]) / len(df_sv["answer"])
print(
    f"SV accuracy: {sv_accuracy} - Initial accuracy: {initial_accuracy} - Direct accuracy: {direct_accuracy}"
)
# %%
