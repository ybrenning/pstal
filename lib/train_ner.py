import pickle
from conllulib import CoNLLUReader, Util

if __name__ == "__main__":
    word_tag_pairs = {}
    tag_adjacencies = {}
    tag_counts = {}
    first_tags = {}

    with open("../sequoia/sequoia-ud.parseme.frsemcor.simple.small", encoding="UTF-8") as f:
        S = sum(1 for line in f if line.strip() == '')

    reader = CoNLLUReader(
        open(
            "../sequoia/sequoia-ud.parseme.frsemcor.simple.small",
            encoding="UTF-8"
        )
    )

    for sent in reader.readConllu():
        bio = CoNLLUReader.to_bio(sent)

        # Count word, tag occurences
        for word, tag in zip(sent, bio):
            word = str(word)
            if (word, tag) in word_tag_pairs.keys():
                word_tag_pairs[(word, tag)] += 1
            else:
                word_tag_pairs[(word, tag)] = 1

        # Count tag, tag adjacencies
        for i in range(len(bio) - 1):
            t1 = bio[i]
            t2 = bio[i + 1]
            if (t1, t2) in tag_adjacencies.keys():
                tag_adjacencies[(t1, t2)] += 1
            else:
                tag_adjacencies[(t1, t2)] = 1

        # Count tag occurences
        for tag in bio:
            if tag in tag_counts.keys():
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

        # Count first tags
        if bio[0] in first_tags.keys():
            first_tags[bio[0]] += 1
        else:
            first_tags[bio[0]] = 1

        E = {}
        for k, v in word_tag_pairs.items():
            E[k] = Util.log_cap(tag_counts[k[1]]) - Util.log_cap(v)

        T = {}
        for k, v in tag_adjacencies.items():
            T[k] = Util.log_cap(tag_counts[k[0]]) - Util.log_cap(v)

        pi = {}
        for k, v in first_tags.items():
            pi[k] = Util.log_cap(S) - Util.log_cap(v)

    with open("E.pickle", "wb") as file_E:
        pickle.dump(E, file_E)

    with open("T.pickle", "wb") as file_T:
        pickle.dump(T, file_T)

    with open("pi.pickle", "wb") as file_pi:
        pickle.dump(pi, file_pi)
