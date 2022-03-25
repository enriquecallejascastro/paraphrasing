# Loading libraries
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_splitter import SentenceSplitter, split_text_into_sentences


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# Paraphrasing a single sentence
def paraphrasing_sentence(input_text,num_return_sequences):
    '''
    Returns n equivalent sentences of a string

    Args:
    - input_text: the sentence we want to paraphrase
    - num_return_sequences: number of paraphrased sentence to generate
    '''
    batch = tokenizer.__call__([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


# Paraphrasing a paragraph
def paraphrasing_paragraph(input_text):
    '''
    Returns a paraphrased paragraph

    Args:
    - input_text: the paragraph we want to paraphrase
    - num_return_sequences: number of paraphrased sentence to generate
    '''
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(input_text)
    paraphrase = []
    for i in sentence_list:
        a = paraphrasing_sentence(i,1)
        paraphrase.append(a)
    paraphrase2 = [' '.join(x) for x in paraphrase]
    paraphrase3 = [' '.join(x for x in paraphrase2) ]
    paraphrased_text = str(paraphrase3).strip('[]').strip("'")
    return paraphrased_text
    