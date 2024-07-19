import nnet
import os

# Params
workers_prepare = -1 # Set to -1 for nproc
mean_face_path = "media/20words_mean_face.npy"
tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"

lrs2_username = "lrs2008" # Set to your lrs2 username
lrs2_password = "et7uonoh" # Set to your lrs2 password

print("Download and Prepare LRS2")
os.environ["LRS2_USERNAME"] = lrs2_username
os.environ["LRS2_PASSWORD"] = lrs2_password
lrs2_dataset = nnet.datasets.LRS(None, None, version="LRS2", download=True, prepare=True, tokenizer_path=tokenizer_path, mean_face_path=mean_face_path, workers_prepare=workers_prepare, mode="pretrain+train+val")

print("Create Corpora")
lrs2_dataset.create_corpus(mode="pretrain")
lrs2_dataset.create_corpus(mode="train")
lrs2_dataset.create_corpus(mode="val")
lrs2_dataset.create_corpus(mode="test")


filenames = ["datasets/LRS2/corpus_pretrain.txt", "datasets/LRS2/corpus_train.txt", "datasets/LRS2/corpus_val.txt"]
with open("datasets/LRS2/corpus_lrs2_pretrain+train+val.txt", "w") as fw:
    for filename in filenames:
        if os.path.exists(filename): 
            with open(filename, "r") as fr:
                for line in fr.readlines():
                    fw.write(line)
        else:
            print(f"Skipping non-existent file: {filename}")
