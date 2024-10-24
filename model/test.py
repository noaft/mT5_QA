from mt5 import model

context = "Tôi tên Toàn, tôi sinh năm 2003, tôi là học sinh trường đại học UIT, tui có kiến thức khá gà."
question = "Toàn có kiến thức như nào?"

path = "/home/toan03/mT5_QA/checkpoint-12000"
tokenizer = "google/mt5-small"
mT5 = model(path, tokenizer )

r = mT5.answer_question(question, context)
print(r)