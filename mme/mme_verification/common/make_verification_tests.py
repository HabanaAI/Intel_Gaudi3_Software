import sys
import os
import io
import argparse

platforms = ["gaudi2"]

def parse_args():
    default_path = os.environ["MME_ROOT"] + "/tests/mme_tests/mme_verification_tests.cpp"
    parser = argparse.ArgumentParser(description="Adds mme test JSON files to CI tests")
    all_platforms = platforms + ["all"]
    parser.add_argument('--platform', dest='platform', choices=all_platforms,
                        help="Platform used - gaudi2", default="all", required=False)
    parser.add_argument('--path', dest='path', default=default_path, help="Path to cpp test file", required=False)
    return parser.parse_args()


def get_test_class_name(platform):
    if platform == "gaudi2":
        return "MMEGaudi2Verification"
    else:
        raise Exception("unknown platform")


def add_test(platform, test_name, fp):
    test_class = get_test_class_name(platform)
    test_dec = "TEST_F(" + test_class + ", " + test_name + ")\n"
    fp.write("\n"
             + test_dec +
             "{\n"
             "    runTest(test_info_->name());\n"
             "}\n")


def write_header(fp):
    fp.write("""#include "mme_test_base.h"
     
    """)

def write_tests_for_platform(platform, fp):
    directory_name = os.environ["MME_ROOT"] + "/mme_verification/" + platform + "/configs/"

    for filename in os.listdir(directory_name):
        if filename.endswith(".json"):
            test_name = os.path.basename(filename)
            test_name = os.path.splitext(test_name)[0]
            if test_name == "config_all":
                continue

            print(os.path.join(directory_name, filename))
            add_test(platform, test_name, fp)
        else:
            continue



def main():
    args = parse_args()
    with open(args.path, "w") as fp:
        write_header(fp)
        if args.platform == "all":
            for platform in platforms:
                write_tests_for_platform(platform, fp)
        else:
            write_tests_for_platform(args.platform, fp)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Failed! {e}")
        sys.exit(1)

