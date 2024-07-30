import datasets
from datasets import Dataset, load_dataset

def conll_to_dict(file):

    labels_dict = dict(
        O=0,
        Bempresa=1,
        Iempresa=2,
        Bempresario=3,
        Iempresario=4,
        Bpolitico=5,
        Ipolitico=6,
        Boutras_pessoas=7,
        Ioutras_pessoas=8,
        Bvalor_financeiro=9,
        Ivalor_financeiro=10,
        Bcidade=11,
        Icidade=12,
        Bestado=13,
        Iestado=14,
        Bpais=15,
        Ipais=16,
        Borganização=17,
        Iorganização=18,
        Bbanco=19,
        Ibanco=20
    )

    words = []
    labels_list = []
    ner_tags = []

    words_atual = []
    labels_atual = []
    ner_tags_atual = []

    i = 0

    with open(file) as f:
        for line in f:
            if "-DOCSTART-" in line:
                continue
            if line == '\n':
                if words_atual:
                    words.append(list(words_atual))
                    labels_list.append(list(labels_atual))
                    ner_tags.append(list(ner_tags_atual))
                    words_atual.clear()
                    labels_atual.clear()
                    ner_tags_atual.clear()
                continue
            else:
                text = line.split()
                words_atual.append(text[0])
                labels_atual.append(text[3])
                tag = text[3].replace('-', '')
                ner_tags_atual.append(labels_dict[tag])

    raw_data_dict = {}
    for i in range(len(words)):
        raw_data_dict[i] = {}
        raw_data_dict[i]['words'] = list(words[i])
        raw_data_dict[i]['original_labels'] = list(labels_list[i])
        raw_data_dict[i]['ner_tags'] = list(ner_tags[i])


    # converting to a list of dictionaries
    # Convert raw_data to a list of dictionaries
    data_list = []
    for idx, data in raw_data_dict.items():
        data_list.append({
            'id': idx,
            'words': data['words'],
            'ner_tags': data['ner_tags'],
            'pos_tags': data['original_labels'],
            'chunk_tags': []  # Placeholder, as your data doesn't have chunk_tags
        })
    return data_list

