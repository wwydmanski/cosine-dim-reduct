import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
from io import StringIO

DEVICE = "cuda:0"
model = SentenceTransformer('all-mpnet-base-v2').to(DEVICE)


def _embed2(term, device):
    sentence_rep = model.to(device).encode(term)
    return sentence_rep

def embed(term):
    try:
        outputs = _embed2(term, DEVICE)
    except RuntimeError as e:
        print("Switching to CPU! Reason: " + str(e))
        outputs = _embed2(term, "cpu")
    return outputs

def prepare_sentences(line):
    res = []
    
    prepped = json.loads(line)
    res.append(prepped['query'])
    res.append(prepped['pos'])        
    res.append(prepped['neg'])
    return res

def save_vectors(vectors):
    vectors = np.array(vectors)
    np.save("data/vectors.npy", vectors, allow_pickle=False)
    
    try:
        with open("data/meta.txt", "r") as f:
            prev = f.read().split()[-1]
            prev = prev.split(",")[0]
            prev = int(prev)
    except IndexError:
        prev = -1
        
    current = prev+1
    with open("data/meta.txt", "a") as f:
        for _ in vectors[::3]:
            f.write(f"{current},anchor\n")
            f.write(f"{current},positive\n")
            f.write(f"{current},negative\n")
            current += 1
            
            
if __name__ == "__main__":
    with open("data/meta.txt", "w") as f:
        pass

    TOTAL = 100000
    buffer = []
    all_embeddings = []
    with open("data/marco_small.jsonl", "r") as f:
        for i, line in enumerate(tqdm.tqdm(f, total=TOTAL)):
            buffer.extend(prepare_sentences(line))

            if len(buffer) < 128:
                continue

            embeddings = embed(buffer)
            all_embeddings.extend(embeddings.tolist())
            buffer = []
            
    save_vectors(all_embeddings)