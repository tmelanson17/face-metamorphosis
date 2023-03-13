
def get_index_pairs():
    index_pairs = list()
    with open("facial-regions.txt", "r") as f:
        entries = f.readlines()
        for entry in entries:
            begin_str, end_str, _ = entry.split(',')
            begin = int(begin_str)
            end = int(end_str)
            for i in range(begin, end):
                index_pairs.append((i-1, i))
    return index_pairs

if __name__ == '__main__':
    _index_pairs = get_index_pairs()
    for pair in _index_pairs:
        print(pair)
