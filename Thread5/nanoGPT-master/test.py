import json
import matplotlib.pyplot as plt
import numpy as np

with open('val_losses_a2r0.json', 'r') as file:
    val_losses_a2r0 = json.load(file)

val_losses_a2r0 = val_losses_a2r0[:200]

print(len(val_losses_a2r0))