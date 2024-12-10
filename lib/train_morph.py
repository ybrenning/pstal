import numpy as np
from conllulib import CoNLLUReader


def process_sent(sent, vocab_in, vocab_out):

    id_in = max(list(vocab_in.values()))
    id_out = max(list(vocab_out.values()))

    end = 0
    chars = "<pad>"
    in_enc = []
    out_enc = []
    ends = []

    for tok in sent:
        if tok["feats"] and "Number" in tok["feats"].keys():
            num = tok["feats"]["Number"]
            if num not in vocab_out.keys():
                id_out += 1
                vocab_out[num] = id_out
            out_enc.append(vocab_out[num])

        for c in str(tok):
            if c not in vocab_in.keys():
                id_in += 1
                vocab_in[c] = id_in
            chars += c
            in_enc.append(vocab_in[c])
            end += 1
        ends.append(end)

        chars += "<esp>"
        end += 1

    # Note: replacing final <esp> with <pad>
    chars = chars[:-5] + "<pad>"

    return chars, np.array(in_enc), np.array(ends), np.array(out_enc)


def pad(arr, m, p):
    if len(arr) > m:
        return arr[:m]
    else:
        padding = [p] * (m - len(arr))
        return np.concatenate([arr, padding])


if __name__ == "__main__":
    reader = CoNLLUReader(
        open(
            "../sequoia/sequoia-ud.parseme.frsemcor.simple.small",
            encoding="UTF-8"
        )
    )

    vocab_in = {"<pad>": 0, "<unk>": 1, "<esp>": 2}
    vocab_out = {"<pad>": 0, "<N/A>": 1}

    for sent in reader.readConllu():
        chars, in_enc, ends, out_enc = process_sent(sent, vocab_in, vocab_out)

        in_enc = pad(in_enc, m=200, p=0)
        ends = pad(ends, m=20, p=0)
        out_enc = pad(out_enc, m=20, p=0)

        print(in_enc)
        print(ends)
        print(out_enc)
        assert 0
