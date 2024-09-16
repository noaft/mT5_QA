from mt5 import model

context = "Tôi tên Toàn, tôi sinh năm 2003, tôi là học sinh trường đại học UIT, tui có kiến thức khá gà."
question = "Toàn có kiến thức như nào?"

model_name = "google/mt5-small"
tokenizer = "google/mt5-small"
mT5 = model(model_name = model_name, tokenizer = tokenizer )