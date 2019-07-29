import os

metadata_path = 'metadata_sm.csv'
folder = 'filenames_sm'

data_path = '/workspace/data/aws/dataset/samantha/wavs/'

low_filter = 2
up_filter = 30

filtered = 0
passed = 0

lines = []

with open(metadata_path, 'r') as f:
    for l in f:
        splt = l.strip().split('|')
        sentence = splt[1]

        c = sentence.count(' ')

        if c >= low_filter and c <= up_filter:
            passed += 1

            lines.append('|'.join(['{}{}.wav'.format(data_path, splt[0]), sentence]) + '\n')
        else:
            filtered += 1

print('Passed:', passed)
print('Filtered:', filtered)


if lines:
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)

    train_amount = int(len(lines) * 0.95)
    val_amount = int(len(lines) * 0.025)
    test_amount = len(lines) - train_amount - val_amount

    train = lines[:train_amount]
    val = lines[train_amount:train_amount+val_amount]
    test = lines[-test_amount:]

    with open(os.path.join(folder, 'train.txt'), 'w') as f:
        f.writelines(lines)

    # with open(os.path.join(folder, 'val.txt'), 'w') as f:
    #     f.writelines(val)
    #
    # with open(os.path.join(folder, 'test.txt'), 'w') as f:
    #     f.writelines(test)




