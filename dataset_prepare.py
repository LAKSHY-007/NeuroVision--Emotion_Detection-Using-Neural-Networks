import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def atoi(s):
    return sum((ord(c) - ord('0')) * 10**i for i, c in enumerate(reversed(s)))

def create_direc(base_path='data'):
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
    
    for split in ['train', 'test']:
        for emotion in emotions:
            Path(base_path, split, emotion).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory structure under {base_path}")

def process_fer2013_dataset(csv_path='./fer2013.csv', output_dir='data'):
    counters = {
        'train': {emotion: 0 for emotion in range(7)},
        'test': {emotion: 0 for emotion in range(7)}
    }
    
    emotion_to_folder = {
        0: 'angry',
        1: 'disgusted',
        2: 'fearful',
        3: 'happy',
  