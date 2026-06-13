# module dd


import pprint
import sys

def dd(*args):
    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")
    print("Dumping data:")
    print("\n")
    for arg in args:
        pprint.pprint(arg, stream=sys.stdout)

    print("---------------------------------------")
    print("---------------------------------------")
    print("---------------------------------------")



    raise Exception("Dumping data, stopping execution.")
    