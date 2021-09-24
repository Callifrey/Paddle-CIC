import os

def check_file(file1, file2):
    filename1 = os.path.split(os.path.splitext(file1)[0])[-1]
    filename2 = os.path.split(os.path.splitext(file2)[0])[-1]
    return filename1 == filename2

def get_paths(root):
    paths = []
    dir = os.path.expanduser(root)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                paths.append(path)
    return paths
