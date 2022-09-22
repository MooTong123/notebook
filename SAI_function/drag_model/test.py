# -- coding: utf-8 --



def main():
    x = '1,2,19'

    a = [int(i) for i in x.split(',')]
    print(max(a))


if __name__ == '__main__':
    main()