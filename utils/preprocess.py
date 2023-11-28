import json
import glob
import random

query = []

def preprocess_data_for_classification(path, split):
    global query
    examlpe_data = []
    for file in glob.glob(path + "bert_data.[0-9].json"):
        with open(file, "r", encoding = "utf8") as f:
            groups = json.load(f)
            for group in groups:
                for line in group["candidates"]:
                    examlpe_data += [{
                        "query_idx": len(query), 
                        "cypher": line[0], 
                        "label": line[1]
                    }] * (19 * int(line[1] == 1 and split == "train") + 1)
                    query += [group["query"]]
    random.shuffle(examlpe_data)
    return examlpe_data

train_path = "./bert_data/train/"
eval_path = "./bert_data/eval/"
train = preprocess_data_for_classification(train_path, "train")
eval = preprocess_data_for_classification(eval_path, "eval")
with open("./bert_data/train/train.json", "w", encoding="utf8") as f:
    json.dump(train, f, indent=4, ensure_ascii=False)
with open("./bert_data/eval/eval.json", "w", encoding="utf8") as f:
    json.dump(eval, f, indent=4, ensure_ascii=False)
with open("./bert_data/query.json", "w", encoding="utf8") as f:
    json.dump(query, f, indent=4, ensure_ascii=False)