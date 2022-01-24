import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils import config
from process_data import process_data


if __name__ == '__main__':
    # STEP 1: PROCESS AND CLEAN THE DATA
    df = process_data()