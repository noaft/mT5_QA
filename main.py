from model.mt5 import model as MT5


if __name__ == "__main__":
    model_name = "google/mt5-small"
    tokenizer = "google/mt5-small"
    model = MT5(model_name = model_name, tokenizer = tokenizer )