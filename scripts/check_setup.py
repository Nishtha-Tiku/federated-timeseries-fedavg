from src.config import Config
from src.data import load_data
from src.model import LSTMClassifier
from src.utils import set_seed

def main():
    cfg=Config(); set_seed(cfg.seed)
    (x_train,y_train,subjects_train),(x_test,y_test,subjects_test)=load_data()
    model=LSTMClassifier(input_size=x_train.shape[-1],hidden_size=cfg.hidden_size,num_layers=cfg.num_layers)
    print("\n=== Setup check ==="); print("Device:",cfg.device); print("Train shape:",tuple(x_train.shape)); print("Test shape:",tuple(x_test.shape)); print("Classes:",len(set(y_train.tolist()))); print("Train subjects:",len(set(subjects_train.tolist()))); print("Test subjects:",len(set(subjects_test.tolist()))); print("Sensor channels:",x_train.shape[-1]); print("Timesteps:",x_train.shape[1]); print("Model parameters:",f"{sum(p.numel() for p in model.parameters()):,}"); print("Setup check completed successfully.")
if __name__=="__main__": main()
