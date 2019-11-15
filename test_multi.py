from multiprocessing import Pool
class Test:
    def __init__(self):
        self.test = 0
def f(x):
    print('multiprocess', x)
    x.test = 5
    return x

if __name__ == '__main__':
    a = Test()
    with Pool(1) as p:
        print(a)
        a = p.map(f,[a] )
        a=a[0]
        print(a.test)
        print(a)