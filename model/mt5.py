from transformers import AutoTokenizer, MT5ForQustionAnswering
import torch

class config:
    max_length = 384
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    model_name = 'google/mt5-small'
    tokenizer = 'google/mt5-small'
    do_train = True
    train_file = ""
    test_file = ""
    evalute = ""
    

class Model:
    """
    Initialize model and device.
    model_name: Name of the model or path to local model checkpoint
    tokenizer: Tokenizer to split and embed text into tokens
    device: Device to be used (e.g., 'cuda' or 'cpu')
    """
    def __init__(self, model_name, tokenizer_name, device, path_model=None, path_token=None):
        # Initialize model
        self.tokenizer = None
        self.model = None
        
        # Load model from local path if provided
        if path_model:
            self.model = MT5ForQustionAnswering.from_pretrained(path_model)
        else:
            self.model = MT5ForQustionAnswering.from_pretrained(model_name)
        
        # Load tokenizer from local path if provided
        if path_token:
            self.tokenizer = AutoTokenizer.from_pretrained(path_token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set device
        self.device = device
        self.model.to(self.device)

    def predict(self, context, question):
        """
        Predict answer from context and question.
        context: Paragraph containing the answer
        question: Question to find the answer for
        """
        # Tokenize the input
        inputs = tokenizer(question, context, return_tensors="pt")
        # Perform inference (this returns the start and end logits for the answer span)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # Extract start and end logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the most likely start and end token positions
        start_position = torch.argmax(start_logits)
        end_position = torch.argmax(end_logits)

        # Decode the answer tokens
        answer_tokens = inputs["input_ids"][0][start_position:end_position+1]
        answer = tokenizer.decode(answer_tokens)
        returns answer



    def trainning(self, data):
        """
        Trainning data with data fomated same like Squad
        data:{
            data:{
                context{
                .....
                },
                qa{
                    qas: question here
                    ,
                    ans:{
                        answer: answer here,
                        start position: position first text in paragraphs
                    }
                }
            }
        }
        """
        pass