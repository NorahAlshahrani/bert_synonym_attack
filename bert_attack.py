from transformers import logging as hflogging
import logging, warnings, os, tensorflow as tf
from huggingface_hub.utils import disable_progress_bars


disable_progress_bars()
tf.autograph.set_verbosity(0)
hflogging.set_verbosity_error()
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)



def load(model, dataset):
    if model == 'bert' and dataset == 'hard':
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
        model = BertForSequenceClassification.from_pretrained('NorahAlshahrani/BERThard', num_labels=4)
        return model, tokenizer

    elif model == 'cnn' and dataset == 'hard':
        import pickle
        from huggingface_hub import from_pretrained_keras
        tokenizer = pickle.load(open('tokenizerCNNhard.pickle', 'rb'))
        model = from_pretrained_keras('NorahAlshahrani/2dCNNhard')
        return model, tokenizer

    elif model == 'bilstm' and dataset == 'hard':
        import pickle
        from huggingface_hub import from_pretrained_keras
        tokenizer = pickle.load(open('tokenizerbiLSTMhard.pickle', 'rb'))
        model = from_pretrained_keras('NorahAlshahrani/biLSTMhard')
        return model, tokenizer

    elif model == 'bert' and dataset == 'msda':
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
        model = BertForSequenceClassification.from_pretrained('NorahAlshahrani/BERTmsda', num_labels=3)
        return model, tokenizer

    elif model == 'cnn' and dataset == 'msda':
        import pickle
        from huggingface_hub import from_pretrained_keras
        tokenizer = pickle.load(open('tokenizerCNNmsda.pickle', 'rb'))
        model = from_pretrained_keras('NorahAlshahrani/2dCNNmsda')
        return model, tokenizer

    elif model == 'bilstm' and dataset == 'msda':
        import pickle
        from huggingface_hub import from_pretrained_keras
        tokenizer = pickle.load(open('tokenizerbiLSTMmsda.pickle', 'rb'))
        model = from_pretrained_keras('NorahAlshahrani/biLSTMmsda')
        return model, tokenizer

    else:
        print("ERROR: load() function takes 2 arguments: \n  \
        model={bert, cnn, or bilstm}, \n\t  dataset={hard or msda}")



def predict(text, model, dataset):
    import numpy as np
    if model == 'bert' and dataset == 'hard':
        model, tokenizer = load('bert', 'hard')
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        outputs = model(**inputs).logits
        id2label = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        predicted_class_id = outputs.argmax().item()
        preds = outputs.softmax(dim=-1).tolist()
        predicted_score = np.max(preds)
        return id2label[predicted_class_id], predicted_score

    elif model == 'cnn' and dataset == 'hard':
        import torch, numpy as np, tensorflow as tf
        from keras_preprocessing.sequence import pad_sequences
        model, tokenizer = load('cnn', 'hard')
        inputs = tokenizer.texts_to_sequences([text])
        inputs = pad_sequences(inputs, maxlen=512)
        outputs = torch.from_numpy(model.predict(inputs, verbose=0))
        id2label = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        predicted_class_id = outputs.argmax().item()
        preds = outputs.softmax(dim=-1).tolist()
        predicted_score = np.max(preds)
        return id2label[predicted_class_id], predicted_score

    elif model == 'bilstm' and dataset == 'hard':
        import torch, numpy as np, tensorflow as tf
        from keras_preprocessing.sequence import pad_sequences
        model, tokenizer = load('bilstm', 'hard')
        inputs = tokenizer.texts_to_sequences([text])
        inputs = pad_sequences(inputs, maxlen=512)
        outputs = torch.from_numpy(model.predict(inputs, verbose=0))
        id2label = {0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Excellent'}
        predicted_class_id = outputs.argmax().item()
        preds = outputs.softmax(dim=-1).tolist()
        predicted_score = np.max(preds)
        return id2label[predicted_class_id], predicted_score

    elif model == 'bert' and dataset == 'msda':
        model, tokenizer = load('bert', 'msda')
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs).logits
        id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_class_id = outputs.argmax().item()
        preds = outputs.softmax(dim=-1).tolist()
        predicted_score = np.max(preds)
        return id2label[predicted_class_id], predicted_score

    elif model == 'cnn' and dataset == 'msda':
        import torch, numpy as np, tensorflow as tf
        from keras_preprocessing.sequence import pad_sequences
        model, tokenizer = load('cnn', 'msda')
        inputs = tokenizer.texts_to_sequences([text])
        inputs = pad_sequences(inputs, maxlen=330)
        outputs = torch.from_numpy(model.predict(inputs, verbose=0))
        id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_class_id = outputs.argmax().item()
        preds = outputs.softmax(dim=-1).tolist()
        predicted_score = np.max(preds)
        return id2label[predicted_class_id], predicted_score

    elif model == 'bilstm' and dataset == 'msda':
        import torch, numpy as np, tensorflow as tf
        from keras_preprocessing.sequence import pad_sequences
        model, tokenizer = load('bilstm', 'msda')
        inputs = tokenizer.texts_to_sequences([text])
        inputs = pad_sequences(inputs, maxlen=330)
        outputs = torch.from_numpy(model.predict(inputs, verbose=0))
        id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_class_id = outputs.argmax().item()
        preds = outputs.softmax(dim=-1).tolist()
        predicted_score = np.max(preds)
        return id2label[predicted_class_id], predicted_score

    else:
        print("ERROR: predict() function takes 3 arguments: \n  \
        text={str}, \n\t  model={bert, cnn, or bilstm}, \n\t  dataset={hard or msda}")



def tokenize(text):
    tokenized_text = []
    from nltk.tokenize import WhitespaceTokenizer
    tokenized_tokens = WhitespaceTokenizer().tokenize(text)
    for token in tokenized_tokens:
        if token not in tokenized_text:
            tokenized_text.append(token)
    return tokenized_text



def clean(text):
    import re, string, emoji, nltk, ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt', quiet=True)

    from nltk import word_tokenize
    nltk.download('stopwords', quiet=True)
    stop = set(nltk.corpus.stopwords.words("arabic"))

    arabicPunctuations = [".","`","؛","<",">","(",")","*","&","^","%","]","[",",","–",
                          "ـ","،","/",":","؟",".","'","{","}","~","|","!","”","…","“"]


    def remove_punctuation(text):
        cleanText = ''
        for i in text:
            if i not in arabicPunctuations:
                cleanText = cleanText + '' + i
        return cleanText


    def remove_emoji(text):
        emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"
                                       u"\U0001F300-\U0001F5FF"
                                       u"\U0001F680-\U0001F6FF"
                                       u"\U0001F1E0-\U0001F1FF"
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text


    def remove_stopwords(text):
        temptext = word_tokenize(text)
        text = " ".join([w for w in temptext if not w in stop and len(w) >= 2])
        return text


    def remove_noise(text):
        text = re.sub('\s+', ' ', text)
        text = re.sub("\d+", " ", text)
        text =re.sub(r'[a-zA-Z?]', '', text).strip()
        return text


    text = remove_noise(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = remove_punctuation(text)

    return text



def score(text, model, dataset):
    tokenized_cleaned_example = tokenize(clean(text))
    tokenized_original_example = tokenize(text)

    original_example_label, original_example_score = predict(text, model, dataset)

    most_important_words = {}
    less_important_words = {}
    for cln_word in range(0, len(tokenized_cleaned_example)):

        for org_word in tokenized_original_example:

            if tokenized_cleaned_example[cln_word] == org_word:
                idx_org_word = tokenized_original_example.index(org_word)
                tokenized_original_example.remove(org_word)

                new_example = " ".join(tokenized_original_example)
                new_example_label, new_example_score = predict(new_example, model, dataset)

                tokenized_original_example.insert(idx_org_word, org_word)

                if original_example_label != new_example_label:
                    score = original_example_score-new_example_score
                    most_important_words[org_word] = score

                if original_example_label == new_example_label:
                    score = original_example_score-new_example_score
                    less_important_words[org_word] = score

    most_important_words_sorted = sorted(most_important_words.items(), key=lambda x: x[1], reverse=True)

    most_important_words_ordered = []
    for most_importnat_word in most_important_words_sorted:
        most_important_words_ordered.append(most_importnat_word[0])

    less_important_words_sorted = sorted(less_important_words.items(), key=lambda x: x[1], reverse=True)

    less_important_words_ordered = []
    for less_importnat_word in less_important_words_sorted:
        less_important_words_ordered.append(less_importnat_word[0])

    return most_important_words_ordered+less_important_words_ordered



def mask(text, targeted_word, top_k):
    import re
    from transformers import pipeline

    mask_token = '[MASK]'
    text = re.sub(targeted_word, mask_token, text)

    unmasker = pipeline('fill-mask', model='aubmindlab/bert-base-arabertv02', device=-1)
    results = unmasker(text, top_k=top_k)

    synonym_words = []
    for result in results:
        try:
            # print(result['token_str'], end=', ')
            synonym_words.append(result['token_str'])
        except TypeError:
            continue

    return synonym_words



def tag(text):
    from transformers import pipeline
    tagger = pipeline('token-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-ca-pos-egy', device=-1)

    results = tagger(text)
    return results



def check(text1, text2):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
    encoded_text1 = model.encode(text1)
    encoded_text2 = model.encode(text2)

    threshold = 0.80
    similarity = cosine_similarity([encoded_text1], [encoded_text2])[0][0]

    if similarity >= threshold:
        return similarity
    else:
        return None



def sample(model, dataset, min_length, max_length, number_examples):
    import pandas as pd


    def character_length(example):
        words = example.split()
        character_length = sum(len(word) for word in words)
        return character_length


    def has_duplicates(text):
        from nltk.tokenize import WhitespaceTokenizer
        tokenized_tokens = WhitespaceTokenizer().tokenize(text)
        return any(tokenized_tokens.count(token) > 1 for token in tokenized_tokens)


    def select(dataframe, min_length, max_length):
        selected_examples_texts = []
        selected_examples_labels = []
        total_selected_samples = 0

        for i in range(dataframe.shape[0]):
            if character_length(dataframe.iloc[i][1]) >= min_length and \
            character_length(dataframe.iloc[i][1]) <= max_length and \
            has_duplicates(dataframe.iloc[i][1]) == False:
                selected_examples_texts.append(dataframe.iloc[i][1].replace('\n',''))
                selected_examples_labels.append(dataframe.iloc[i][0])
                total_selected_samples+= 1

        new_dataframe = pd.DataFrame(selected_examples_texts, columns=['original_example'])
        new_dataframe['truth_label'] = selected_examples_labels

        return total_selected_samples, new_dataframe


    if dataset == 'hard':
        def condition(x):
            if x==1:
                return 'Poor'
            elif x==2:
                return 'Fair'
            elif x==4:
                return 'Good'
            else:
                return 'Excellent'
            return None

        dataframe = pd.read_csv('HARD-reviews.tsv',sep='\t', header = 0 , encoding = 'utf-16')
        dataframe = dataframe.drop(['nights','room type','user type','Hotel name','no'],axis=1)
        dataframe['rating'] = dataframe['rating'].apply(condition)

        total_selected_samples, dataframe = select(dataframe, min_length, max_length)

        new_dataframe = dataframe.sample(number_examples)

        print("      # Total Number of Qualified Examples:", format(total_selected_samples, ',d'))
        print("      # Total Number of Randomly Selected Examples:", format(new_dataframe.shape[0], ',d'))

        csvfile = f'{model}_{dataset}_selected_samples.csv'
        new_dataframe.to_csv(csvfile, index=False)

        return new_dataframe, csvfile

    elif dataset == 'msda':
        def condition(x):
            if x== 'neg':
                return 'Negative'
            elif x=='neu':
                return 'Neutral'
            else:
                return 'Positive'
            return None

        dataframe = pd.read_csv('Sentiment_Anaysis.csv')
        dataframe = dataframe.drop(['Unnamed: 0'],axis=1)
        dataframe['label'] = dataframe['label'].apply(condition)
        dataframe = dataframe[['label', 'Twits']]

        total_selected_samples, dataframe = select(dataframe, min_length, max_length)

        new_dataframe = dataframe.sample(number_examples)

        print("      # Total Number of Qualified Examples:", format(total_selected_samples, ',d'))
        print("      # Total Number of Randomly Selected Examples:", format(new_dataframe.shape[0], ',d'))

        csvfile = f'{model}_{dataset}_selected_samples.csv'
        new_dataframe.to_csv(csvfile, index=False)

        return new_dataframe, csvfile

    else:
        print("ERROR: select() function takes 4 arguments: \n  \
        dataset={hard or msda}, \n\t  min_length={int}, \n\t  max_length={int} \n\t  number_examples={int}")



def export(dataframe):
    selected_samples_texts = []
    selected_samples_texts = dataframe.original_example.values.tolist()
    return selected_samples_texts



def infer(csvfile, model, dataset):
    import pandas as pd
    dataframe = pd.read_csv(csvfile)

    dataframe['predication_label'] = dataframe['original_example'].apply(lambda example: predict(example, model, dataset)[0])
    dataframe['predication_score'] = dataframe['original_example'].apply(lambda example: predict(example, model, dataset)[1])

    dataframe['inference_results'] = dataframe.apply(lambda x: 'match' if x['truth_label'] == x['predication_label'] else 'mismatch', axis=1)

    accuracy_before_attack__labels = round((dataframe[dataframe['inference_results']=='match'].shape[0]/dataframe.shape[0])*100, 2)
    accuracy_before_attack__scores = round((dataframe.predication_score.mean())*100, 2)

    dataframe.to_csv(f'{csvfile.split(".")[0]}_inference.csv', index=False)

    return dataframe, accuracy_before_attack__labels, accuracy_before_attack__scores



def attack(examples, model, dataset):
    import pandas as pd
    all_results = pd.DataFrame(columns=['model', 'dataset', 'example', 'predication_label', 'predication_score',
                                        'targeted_word', 'synonym_word', 'adversarial_example', 'adversarial_label', 'adversarial_score'])

    successful_attacks = pd.DataFrame(columns=['model', 'dataset', 'example', 'predication_label', 'predication_score',
                                               'targeted_word', 'synonym_word', 'adversarial_example', 'adversarial_label', 'adversarial_score'])

    for example in examples:

        predication_label, predication_score = predict(example, model, dataset)
        idx_words_example = tokenize(example)

        tagged_original_example = tag(example)

        important_words = score(example, model, dataset)

        for word in important_words:
            targeted_word = word
            synonym_words = mask(example, targeted_word, 10)

            idx_word = idx_words_example.index(word)
            word_pos = tagged_original_example[idx_words_example.index(idx_words_example[idx_word])]['entity']

            for s in range(0, len(synonym_words)):

                if word != synonym_words[s]:

                    idx_replaced_words_example = list(idx_words_example)
                    idx_replaced_words_example[idx_word] = synonym_words[s]

                    adversarial_example = " ".join(idx_replaced_words_example)

                    idx_words_adversarial_example = tokenize(adversarial_example)
                    adversarial_targeted_word = synonym_words[s]
                    tagged_adversarial_example = tag(adversarial_example)
                    synonym_word_pos = tagged_adversarial_example[idx_words_adversarial_example.index(adversarial_targeted_word)]['entity']

                    if word_pos == synonym_word_pos:
                        similarity = check(example, adversarial_example)

                        if similarity != None:

                            adversarial_label, adversarial_score = predict(adversarial_example, model, dataset)

                            all_results = all_results.append({'model':model,
                                                              'dataset':dataset, 'example':example,
                                                              'predication_label':predication_label, 'predication_score':predication_score,
                                                              'targeted_word':targeted_word,  'synonym_word':synonym_words[s],
                                                              'adversarial_example':adversarial_example,
                                                              'adversarial_label':adversarial_label, 'adversarial_score':adversarial_score}, ignore_index = True)

                            if predication_label == adversarial_label:
                                continue

                            if predication_label != adversarial_label:
                                print("        # Original Example:", example)
                                print("          > Predication Label:", predication_label)
                                print("          > Predication Score:", predication_score)
                                print("            * Targeted Word:", targeted_word)
                                print("            * Synonym Word:", synonym_words[s])
                                print("        ----------------------------------------------")
                                print("        ***** SUCCESSFUL ATTACK FOUND & RECORDED *****")
                                print("        ----------------------------------------------")
                                print("        # Adversarial Example:", adversarial_example)
                                print("          < Adversarial Label: ", adversarial_label)
                                print("          < Adversarial Score: ", adversarial_score, "\n")

                                successful_attacks = successful_attacks.append({'model':model,
                                                                        'dataset':dataset, 'example':example,
                                                                        'predication_label':predication_label,
                                                                        'predication_score':predication_score,
                                                                        'targeted_word':targeted_word,  'synonym_word':synonym_words[s],
                                                                        'adversarial_example':adversarial_example, 'adversarial_label':adversarial_label,
                                                                        'adversarial_score':adversarial_score}, ignore_index = True)
                                break

            break

    all_results.to_csv(f'{model}_{dataset}_all_results.csv', index=False)

    csvfile = f'{model}_{dataset}_successful_attacks.csv'
    successful_attacks.to_csv(csvfile, index=False)

    return successful_attacks, csvfile



def compute(csvfile, number_examples, model, dataset):
    import pandas as pd
    dataframe = pd.read_csv(csvfile)

    accuracy_after_attack__labels = round(((number_examples-dataframe.shape[0])/number_examples)*100, 2)
    accuracy_after_attack__scores = round((dataframe.adversarial_score.mean())*100, 2)

    dataframe.to_csv(f'{csvfile.split(".")[0]}_inference.csv', index=False)

    return dataframe, accuracy_after_attack__labels, accuracy_after_attack__scores



def main(models, datasets, min_length, max_length, number_examples):

    for model in models:

        for dataset in datasets:

            print(f"@@@ BEGINNING ATTACK: [ MODEL: `{model.upper()}` | DATASET: `{dataset.upper()}` ] @@@")

            print(f"\n   ## Sampling Examples from `{dataset.upper()}` Dataset:")
            selected_samples_dataframe, selected_samples_csv = sample(model, dataset, min_length, max_length, number_examples)

            _, accuracy_before_attack__labels, accuracy_before_attack__scores = infer(selected_samples_csv, model, dataset)

            selected_samples_texts = export(selected_samples_dataframe)

            print(f"\n   ## Running BERT Synonym-level Attack on `{model.upper()}` Model:")
            examples = list(selected_samples_texts)

            _, successful_attacks_csv = attack(examples, model, dataset)

            _, accuracy_after_attack__labels, accuracy_after_attack__scores = compute(successful_attacks_csv, number_examples, model, dataset)

            print("   ## Attack Results:")
            print("        # Model:", model.upper())
            print("        # Dataset:", dataset.upper())
            print("          > Accuracy Before Attack (via Truth Labels):", accuracy_before_attack__labels, "%")
            print("          > Accuracy Before Attack (via Predication Scores):", accuracy_before_attack__scores, "%")
            print("          < Accuracy After Attack (via Predication Labels):", accuracy_after_attack__labels, "%")
            print("          < Accuracy After Attack (via Predication Scores):", accuracy_after_attack__scores, "%\n\n")

            # break

        # break
