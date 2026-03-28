# %%
# Read the file data/cells.txt which has data like:
# 720575940573484592
# 720575940557128920
# ...
# Then send to clipboard the comma-separated values of the numbers in the file.
import pyperclip

with open("data/cells.txt", "r") as f:
    data = f.read().strip().splitlines()
csv_data = ",".join(data)

pyperclip.copy(csv_data)
print("Comma-separated values copied to clipboard.")
# %%
