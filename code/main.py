from paraphrasing_generation import paraphrasing_paragraph, paraphrasing_sentence

# Paraphrasing a single sentence
text = 'We need to schedule new meeting times to include everyone.'
paraphrasing_sentence(text, 5)

# Paraphrasing a paragraph
context = "The highlight of this study is that the same light-weight model trained by keeping the objective of Paraphrase Generation can also be used for solving the Paraphrase Identification task. Hence, the proposed system islight-weight in terms of the model’s size along with the data used to train the model which facilitates the quick learning of the model without having to compromise with the results."
paraphrasing_paragraph(context)
