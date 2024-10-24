import json

def processing_data(file_path, save_data):
  real_data = {
      'data' : []
  }
  f = open(file_path, 'r')
  data_data = json.load(f)
  datas = data_data['data']
  for data in datas:
    title = data['title']
    pras = data['paragraphs']
    for pra in pras:
      context = pra['context']
      qas = pra['qas']
      for qa in qas:
        question = qa['question']
        id = qa['id']
        answers = qa['answers']
        real_data['data'].append({
            'title': title,
            'context': context,
            'question': question,
            'id': id,
            'answers': answers
        })
  with open(save_data, 'w') as f:
    json.dump(real_data, f)
  f.close()
  print('done')
