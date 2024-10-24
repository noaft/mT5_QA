import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer

class model:

    def __init__(self, path, tokenizer):
        self.model = MT5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

    def answer_question(self, question, context ):
        # Chuẩn bị input cho mô hình mT5
        input_text = f"question: {question} context: {context}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate output từ mô hình mT5
        with torch.no_grad():
            output_ids = self.model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

        # Decode output thành câu trả lời
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer
