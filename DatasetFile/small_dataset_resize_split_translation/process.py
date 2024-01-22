a_word = []
q_word = []
a_path = "./a_vocab.txt"
q_path = "./q_vocab.txt"

with open("./train_modified.txt", "r", encoding="GBK") as f:
    for line in f.readlines():
        line = line.strip()
        info = line.split("Q")
        question, gts = info[2], info[1]
        question_words = question.split(" ")
        for word in question_words:
            if word not in q_word:
                q_word.append(word)
        gts_words = gts.split(" ")
        for word in gts_words:
            if word not in a_word:
                a_word.append(word)

with open("./test_modified.txt", "r", encoding="GBK") as f:
    for line in f.readlines():
        line = line.strip()
        info = line.split("Q")
        question, gts = info[2], info[1]
        question_words = question.split(" ")
        for word in question_words:
            if word not in q_word:
                q_word.append(word)
        gts_words = gts.split(" ")
        for word in gts_words:
            if word not in a_word:
                a_word.append(word)

with open("./q_vocab.txt", "w", encoding="GBK") as f:
    for word in q_word:
        f.write(word+"\n")

with open("./a_vocab.txt", "w", encoding="GBK") as f:
    for word in a_word:
        f.write(word + "\n")


print(len(q_word))
print(len(a_word))
