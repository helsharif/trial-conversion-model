from trial_conversion_model.features import build_training_data
from trial_conversion_model.train import train

if __name__ == "__main__":
    build_training_data()
    print(train())
