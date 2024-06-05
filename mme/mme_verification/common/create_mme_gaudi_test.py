#!/usr/bin/env python
# This script reads a set of test demp files from synapse and creates mme tests accordingly
import sys, getopt
import argparse
import glob
import subprocess
import os

def GetArgs(argv):
    parser = argparse.ArgumentParser(description='Create mme gaudi tests')
    parser.add_argument('path', metavar='path',  type=str, help='path to synapse dump files')
    parser.add_argument('op', metavar='op',  type=str, help='Operation: fwd, dedx, dedw, ab, abt, atb, atbt')
    parser.add_argument('num_tests', metavar='num_tests', type=int, default=10, help='number of tests')

    args = parser.parse_args()
    return args.path, args.op, args.num_tests

if __name__ == "__main__":
    path, op, num_tests = GetArgs(sys.argv[1:])

    file1 = open("mmeCfgFile.cfg", "a")

    file1.write("global:pole=north\n")
    file1.write("global:sramBase=0x7ff0080000\n")
    file1.write("global:sramSize=0x1380000\n")
    file1.write("global:hbmBase=0x20000000\n")
    file1.write("global:multiplTests=1\n")
    file1.write("global:fp=1\n")
    file1.write("global:shuffle=0\n")
    file1.write("global:programInSram=0\n")
    file1.write("global:smBase=0x7ffc4f0000\n")
    file1.write("\n")

    i = 0
    while i < num_tests:
        inputName = "gaudi_node_" + op + "_" + str(i) + ".cfg"
        with open(inputName) as f:
            lines = f.readlines()

        file1.write("testName=gaudiTest_" + str(i) + "\n")
        file1.writelines("%s" % l for l in lines)

        file1.write("unrollEn=1\n")
        file1.write("roundingMode = random\n")
        file1.write("stochasticRoundingMode = random\n")
        file1.write("xInSram = 1\n")
        file1.write("yInSram = 1\n")
        file1.write("wInSram = 1\n")
        file1.write("sramStreamingBudget = -1\n")
        file1.write("xMinVal = -10\n")
        file1.write("xMaxVal = 20\n")
        file1.write("yMinVal = -10\n")
        file1.write("yMaxVal = 10\n")
        file1.write("wMinVal = -1\n")
        file1.write("wMaxVal = 1\n")
        file1.write("skipRef = 0\n")
        file1.write("\n")

        i += 1


    file1.close()

