import numpy as np
import pandas as pd

cls_arrays = [np.load(f"cls_tokens_{i}.npy") for i in range(10)] 
cls_concatenation = np.concatenate(cls_arrays, axis=0)
cls_concatenation = cls_concatenation.reshape(6980, 768)
np.savetxt('cls_tokens.csv', cls_concatenation, delimiter=',')