
# run command: python -m pdb debugTest.py
if __name__ == "__main__":
    a = 1
    import pdb
    pdb.set_trace()
    b = 2
    c = a + b
    print (c)