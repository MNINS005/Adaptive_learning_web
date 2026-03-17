import sys
import os
sys.path.append(os.path.abspath("."))

from src.pipeline.training_pipeline import TrainingPipeline
training=TrainingPipeline()
training.run()