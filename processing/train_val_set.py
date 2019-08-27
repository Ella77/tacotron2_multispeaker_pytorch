'''
similar datasets

'''
import random
import io


count = 0
def split_huge_file(file,out1,out2,percentage,seed):
    """Splits a file in 2 given the approximate `percentage` to go in the large file."""
    random.seed(seed)
    count = 0
    with io.open(file, 'r',encoding="UTF-8") as fin, \
            io.open(out1, 'w',encoding="UTF-8") as foutBig, \
            io.open(out2, 'w',encoding="UTF-8") as foutSmall:

        for line in fin:
            #print(line)
            count+=1
            r = random.random()
            if r < percentage:
                foutBig.write(line)
            else:
                foutSmall.write(line)

        print(count)
# path = '..'+'/waveglow/actors/actor.txt'
path = 'final_metadata.txt'
split_huge_file(path,'final_train.txt','final_val.txt',0.8,123)
