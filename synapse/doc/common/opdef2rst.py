# *****************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# Author:
# amittal@habana.ai
# ******************************************************************************

## imports for system utilties
from os import listdir, environ
from os.path import isfile, join, expandvars, basename, isdir, exists
from time import strftime
from sys import argv, path, exit
from subprocess import check_output

## imports related to protobuf
from google.protobuf import text_format

path.append(expandvars("$SYNAPSE_ROOT/doc"))
from protobuf import op_def_pb2

## import for rst infrastructure
from common.rstwriter import RstWriter

## other common imports
from re import findall, finditer
import logging

logger = logging.getLogger()

CURR_YEAR = strftime("%Y")
PERF_LIB_FILE = join(environ["SPECS_EXT_ROOT"], "perf_lib_layer_params.h")
COMPANY_LOGO_PATH = join(environ["SYNAPSE_ROOT"], "doc/common/Habana_Intel_Blue.png")
INTRODUCTION_FILE = "introduction.pbtxt"
SUB_COUNTER = 0  ## global counter for maintaining substitution count in rst file


COPYRIGHT = 'This guide is provided by and is the Confidential Information of Habana Labs, Ltd., an Intel company ("Habana").  \
It is provided to you pursuant to the terms and conditions of a non-disclosure agreement by and between you and Habana and should be treated \
accordingly. All information and data contained in this guide are for informational purposes only, without any commitment on the part of Habana, \
and are not to be considered as an offer for a contract. Habana shall not be liable, in any event, for any claims for damages or any other remedy \
in any jurisdiction whatsoever, whether in an action in contract, tort (including negligence and strict liability) or any other theory of \
liability, whether in law or equity including, without limitation, claims for damages or any other remedy in whatever jurisdiction, and shall \
not assume responsibility for patent infringements or other rights to third parties arising out of or in connection with this datasheet. These \
materials are copyrighted and any unauthorized use of these materials may violate copyright, trademark and other laws. Therefore, no part of \
this publication may be reproduced, photocopied, stored on a retrieval system or transmitted without the express written consent of Habana. Any \
new issue of this manual invalidates previous issues.\nHabana Labs, Ltd. reserves the right to revise this publication and to make changes to \
its content at any time, without obligation to notify any person or entity of such revision changes.\n**Copyright © {0} Habana Labs, Ltd. \
an Intel Company. All rights reserved.**'.format(
    CURR_YEAR
)

INTRODUCTION = "The Synapse library holds a collection of operators commonly found in deep learning \
workloads. The library holds operators mapping to Habana's Matrix Multiplication Engine (MME) and operators that are mapped to the Graph Compiler.\n \
The library can be accessed using Synapse API function :code:`synNodeCreateWithId`. This API function accepts a list of input and output tensors, a \
globally unique identifier (GUID) for the operator in the form of character string and a private C structure holding specific \
operator parameters. The purpose of this guide is to list the officially supported operators, their GUIDs, description of the \
operation, the operators' param structures and the operators' general restrictions.\n"
SPHINX_MODE = "ext"  # flag for comparing compilation mode of documentation
DOC_FOLDER = join(environ.get("SYNAPSE_ROOT"), "doc")
OUTPUT_FOLDER = join(DOC_FOLDER, "_auto_generated_docs")


def create_folder(folder):
    try:
        makedirs(folder)
    except OSError as exc:
        if not (exc.errno == errno.EEXIST and isdir(folder)):
            raise RuntimeError("failed to create results folder")


class OpDef2Rst:
    def __init__(self, doc_name, doc_compilation_mode):
        self.doc_name = doc_name
        self.doc_folder = (
            doc_name  # FIXME: once fixed in auto_generate_test_and_kernels
        )
        self.doc_compilation_mode = doc_compilation_mode
        self.ops_folder = join(DOC_FOLDER, self.doc_folder, "op_def")
        self.img_folder = join(DOC_FOLDER, self.doc_folder, "diagrams")
        self.out_file = join(OUTPUT_FOLDER, "synapse_doc_" + self.doc_name + ".rst")

    """
    Adds appendix section to the documentation.
    It first checks if the appendix folder exists. If yes, then for each file in appendix folder, it reads the content and adds it to the doc.
    """

    def add_appendix(self, rst):
        if exists(self.appendix_folder):
            for f_name in listdir(self.appendix_folder):
                with open(join(self.appendix_folder, f_name), "r") as f:
                    f_content = f.read()
                    appendix_list = f_content.split("#. ")
                    appendix_list.remove("")
                for subject in appendix_list:
                    rst.h2(subject.split("\n")[0])
                    rst.paragraph("\n".join(subject.split("\n")[1:]))

    """
    Adds the category's summary table
    A row in summary table consists of operator name, supported input data types, input dimensions and operator's guid
    """

    def add_category_summary_table(self, rst, operators, category):
        header = self.get_summary_table_header()
        table_data = []
        for operator in operators:
            op = operator.op[0]
            guid_list = op.guid
            if guid_list:
                ## extracts common GUID and GUID_dtype
                if len(guid_list) > 1:
                    op_dtypes = []
                    flag = 1
                    ## if the guids in guid_list only differs in datatype then extracts the common guid from list of guid and stores in op_guid
                    ## For example, if guid_list = ["guid_common_part_f32", "guid_common_part_bf16", "guid_common_part_f16"]
                    ## then op_guid = guid_common_part<dtype>
                    ## However, if there is more difference in guids apart from the datatypes then op_guid = " A list of GUIDs "
                    ## For example, if guid_list = ["guid_common_part_diff1_f32", "guid_common_part_diff2_bf16"]
                    ## then op_guid = " A list of GUIDs "
                    if "_" not in guid_list[0]:
                        flag = 0
                    else:
                        common_guid = guid_list[0][: guid_list[0].rindex("_")].strip()
                    for guid in guid_list:
                        if "_" not in guid:
                            flag = 0
                            break
                        guid_first_half = guid[
                            : guid.rindex("_")
                        ].strip()  # extracts the string till last underscore
                        if guid_first_half == common_guid:
                            op_dtypes.append(guid[guid.rindex("_") + 1 :].strip())
                        else:
                            flag = 0
                            break
                    ## GUID Type: ["guid_common_part_f32", "guid_common_part_bf16", "guid_common_part_f16"]
                    if flag:
                        op_guid = common_guid + "<dtype>"
                    ## GUID Type: ["guid_common_part_diff1_f32", "guid_common_part_diff2_bf16"]
                    else:
                        op_guid = " A list of GUIDs "
                        op_dtypes = ["Refer op desc"]
                ## GUID Type: ["guid_common_part_<f32/bf16/f16>"]
                elif guid_list[0].count("<") == 1:
                    op_guid = guid_list[0].split("<")[0] + "<dtype>"
                    op_dtypes = guid_list[0].split("<")[-1].split(">")[0].split("/")
                ## GUID Type: ["guid_common_part_f32"]
                elif "<" not in guid_list[0]:
                    op_guid = guid_list[0]
                    op_dtypes = [guid_list[0].split("_")[-1]]
                ## GUID Type: ["guid_common_part_<diff1/diff2>_<f32/bf16>"]
                else:
                    op_guid = " A list of GUIDs "
                    op_dtypes = ["Refer op desc"]
                ## extracts dimensions
                if len(op.type) > 0:
                    type_desc = op.type[0].description
                    op_dims = findall(r"\d+", type_desc)
                    dims = "D, ".join([dim for dim in op_dims if dim.isnumeric()])
                    dims += "D"
                else:
                    dims = "N/A"
                row = ["`{}`_".format(op.name.strip()), dims, op_guid]
            else:
                row = ["`{}`_".format(op.name.strip()), "N/A", "N/A"]
            table_data.append(row)
        caption = (
            "Table "
            + str(category[0])
            + ". Summary table for "
            + category[1]
            + " operators"
        )
        if self.doc_compilation_mode != SPHINX_MODE:
            rst.table(
                len(table_data), len(table_data[0]), table_data, header, caption=caption
            )
        else:
            rst.table(
                len(table_data),
                len(table_data[0]),
                table_data,
                header,
                caption=caption,
                isExt=True,
            )

    """
    Checks whether the characters in a pbtxt file are within ascii range or not.
    """

    def check_ascii(self, file_content, fname):
        lines = file_content.splitlines()
        for idx, line in enumerate(lines):
            if any(ord(char) >= 128 for char in line):
                print(
                    fname
                    + ":"
                    + str(idx)
                    + " error: the following text line contains non-ascii characters:\n"
                    + line
                )
                exit(1)

    """
    Checks if a pbtxt file has some equations or not. If yes then reads the equation and its label and construct a dictionary
    Returns a dictionary:
        eq_label_dict{
            key: label
            value: equation
        }
    """

    def construct_equation_label_dict(self, formulas, operator_file):
        eq_label_dict = {}
        if formulas:
            for formula in formulas:
                if formula.label:
                    label = formula.label.strip()
                    if formula.equation:
                        eq = formula.equation
                        if label not in eq_label_dict:
                            eq_label_dict[label] = eq
                        else:
                            logger.error(
                                'Error processing file %s More than one formula having label "%s"',
                                operator_file,
                                label,
                            )
                            exit(1)
                    else:
                        logger.error(
                            'Error processing file %s Formula with label = "%s" without equation',
                            operator_file,
                            label,
                        )
                        exit(1)
                else:
                    logger.error(
                        "Error processing file %s Formula without label", operator_file
                    )
                    exit(1)
        return eq_label_dict

    """
    Converts a one dimensional list into nested list using a recursive call. The level of nesting is determined by the number of dollar symbols
    original_list: one dimensional list
    index: position of element in original_list which will be processed
    curr_level: Level of the last inserted value
    new_list: new nested list created from elements of one dimensional list
    Returns index and new_list
    Example:
        original_list = ["Hell0", "$level1", "$$Level11", "$$Level12", "$$$Level123", "$Level2", "Bello"]
        new_list(after all elements are processed) = ['Hell0', ['level1', ['Level11', 'Level12', ['Level123']], 'Level2'], 'Bello']
    """

    def convert_list_to_nested_list(
        self, original_list, index, curr_level, new_list, operator_file=None
    ):
        ## If index is greater than the length of original list then we have reached the end of list
        if index >= len(original_list):
            return index, new_list
        value = original_list[index].strip()
        ## Each value to be inserted as sub-list begins with the dollar symbol
        ## Number of consecutive dollar symbols in the beginning is used to decide the level of list
        new_level = self.count_first_consecutive_occurences("$", value)
        if new_level - curr_level > 1:
            logger.error(
                "Error processing file %s\n. %s: Trying to create %s level nested list without %s level",
                operator_file,
                value,
                new_level + 1,
                new_level,
            )
            exit(1)
        ## If value needs to inserted as sub-level of previous value then create a new list consiting of that value and call convert_list_to_nested_list
        ## append the sub-list to the new_list
        elif new_level - curr_level == 1:
            index += 1
            index, sub_list = self.convert_list_to_nested_list(
                original_list,
                index,
                curr_level + 1,
                [value[curr_level + 1 :]],
                operator_file,
            )
            new_list.append(sub_list)
        if index >= len(original_list):
            return index, new_list
        value = original_list[index].strip()
        new_level = self.count_first_consecutive_occurences("$", value)
        ## If the value needs to be inserted at the same level then append the value to the new_list and call convert_list_to_nested_list for processing further values
        if new_level - curr_level == 0:
            new_list.append(value[new_level:])
            index += 1
            return self.convert_list_to_nested_list(
                original_list, index, curr_level, new_list, operator_file
            )
        elif curr_level - new_level >= 1:
            return index, new_list

    """
    Counts and returns the number of consecutive times a character appeared in a string for the first time.
    """

    def count_first_consecutive_occurences(self, char, str):
        count = 0
        for ch in str.strip():
            if ch == char:
                count += 1
            elif count != 0:
                break
        return count

    """
    Starts with the self.ops_folder as the starting directory and then traverses the drectory tree recursively.
    If a folder is encountered then that is treated as a category and all the files but the introduction.pbtxt within that folder are assigned to that category.
    However, if a folder contains only subfolders and no files, then it is not treated as a category. The sub-folders will be traversed
    and if they contain files then that sub-folder will be treated as a category. All the introduction.pbtxt are saved in another dictionary.
    Returns three dictionaries:
        category_op_dict{
            key: operator category name
            value: list of file contents belonging to the same category
        }
        category_introduction_dict{
            key: operator category name
            value: introduction
        }
        op_file_dict{
            key: operator name
            value: corresponding pbtxt file path
        }
    """

    def get_all_ops(self):
        directories_to_traverse = [self.ops_folder]
        category_op_dict = {}
        category_introduction_dict = {}
        op_file_dict = {}
        while directories_to_traverse:
            directories_to_traverse.sort()
            directories_to_traverse.reverse()
            current_directory = directories_to_traverse.pop()
            directory_name = basename(current_directory)
            listdir(current_directory).sort()
            ## for all the files and folders in current directory
            for x in sorted(listdir(current_directory)):
                ## if it is a folder and has not been explored yet then add it to directories_to_traverse list
                if isdir(join(current_directory, x)):
                    if join(self.ops_folder, x) not in directories_to_traverse:
                        directories_to_traverse.append(join(current_directory, x))
                ## If it is a file then parses the file content and stores it in a dictionary
                else:
                    fname = join(current_directory, x)
                    op_list = op_def_pb2.OpList()
                    with open(fname, "rt") as f:
                        data = f.read()
                        self.check_ascii(data, fname)
                        try:
                            text_format.Parse(
                                data, op_list, allow_unknown_extension=True
                            )
                        except Exception as e:
                            print(f"Error processing op def file {fname}:")
                            print(e)
                            exit(1)
                            break
                        op_file_dict[op_list.op[0].name] = fname
                        is_serial_num = directory_name[:2].isnumeric()
                        if is_serial_num:
                            directory_name = directory_name[3:]
                        directory_name = directory_name.replace("_", " ").title()
                        if directory_name == "Op Def":
                            directory_name = "GUIDs Implemented by the Graph Compiler"
                        if x == INTRODUCTION_FILE:
                            category_introduction_dict[directory_name] = op_list
                        else:
                            if directory_name in category_op_dict:
                                category_op_dict[directory_name].append(op_list)
                            else:
                                category_op_dict[directory_name] = [op_list]
        return category_op_dict, category_introduction_dict, op_file_dict

    """
    Returns currently checked out git branch name
    """

    def get_branch_version(self):
        gitroot = expandvars("$SYNAPSE_ROOT/.git")
        curr_branch = check_output(
            ["git", "--git-dir", gitroot, "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode("utf8")
        return curr_branch.strip()

    """
    Returns introduction
    """

    def get_introduction(self, op_categories):
        intro_cont = (
            "The library operators are divided into "
            + str(len(op_categories))
            + " groups:\n"
        )
        for category in op_categories:
            intro_cont += "#. " + category.title() + "\n"
        return INTRODUCTION + intro_cont

    """
    Returns the header of summary table for each operator category
    """

    def get_summary_table_header(self):
        summary_table_header = ["Operator", " Dimensionality ", " GUID "]
        return summary_table_header

    """
    Reads the perf_lib_layer_params.h file and creates a dictionary with enum name as key and its content as value.
    All the enums in perf_lib_params.h starts with either "enum" or "typedef enum" and ends with a semicolon ";"
    Returns:
        struct_dict: (dict) kernel name as key and their structure as value
    """

    def parse_perf_lib_enum(self):
        if not isfile(PERF_LIB_FILE):
            print(
                "Warning: {0} does not exist. Please provide valid path!!".format(
                    PERF_LIB_FILE
                )
            )
        else:
            enum_dict = {}
            with open(PERF_LIB_FILE, "r") as in_file:
                for curr_line in in_file:
                    if curr_line.strip().startswith(
                        "enum"
                    ) or curr_line.strip().startswith("typedef enum"):
                        if curr_line.startswith("enum"):
                            enum_name = curr_line.split()[-1]
                        enum = curr_line
                        for sub_line in in_file:
                            enum += sub_line
                            if sub_line.strip().endswith(";"):
                                is_enum_name = match("} *([\w_]* *);", sub_line)
                                if is_enum_name is not None:
                                    enum_name = is_enum_name.group(1)
                                break
                        enum_dict[enum_name] = enum
            return enum_dict

    """
    Reads the perf_lib_layer_params.h file and creates a dictionary with kernel name as key and their structure as value.
    Uses standard matching number of opening and closing braces to parse the structure
    Returns:
        struct_dict: (dict) kernel name as key and their structure as value
    """

    def parse_perf_lib_struct(self):
        if not isfile(PERF_LIB_FILE):
            print(
                "Warning: {0} does not exist. Please provide valid path!!".format(
                    PERF_LIB_FILE
                )
            )
        else:
            struct_dict = {}
            with open(PERF_LIB_FILE, "r") as in_file:
                for curr_line in in_file:
                    ## all the structures in perf_lib_params.h starts with "namespace"
                    if curr_line.strip().startswith("namespace"):
                        kernel_name = curr_line.split()[1].strip()
                        stack = []
                        kernel_struct = curr_line
                        for sub_line in in_file:
                            if (
                                "{" in sub_line
                            ):  ##Assumption:   1.Comments does not contain '{' and '}' symbol
                                ##              2.Scope operator '{' doen not start on the same line as the kernel name
                                stack.append("{")
                            if "}" in sub_line:
                                stack.pop()
                            kernel_struct += sub_line
                            if not stack:
                                break
                        struct_dict[kernel_name] = kernel_struct
            return struct_dict

    def get_attr_values(self, attr):
        values = (
            [v.decode("utf-8") for v in attr.s]
            + list(attr.i)
            + list(attr.f)
            + list(attr.b)
        )
        s = ", ".join([f"{str(v)}" for v in values])
        if len(values) > 1:
            s = "{" + s + "}"
        return s

    def add_attr(self, rst, op_attr):
        for x in op_attr:
            key = "{0} : {1}".format(x.name, x.type)
            allowed_values = self.get_attr_values(x.allowed_values)
            default_values = self.get_attr_values(x.default)
            sep = "("
            if allowed_values:
                key += " (allowed is {0}".format(allowed_values)
                sep = ", "
            if default_values:
                if allowed_values:
                    key += sep + "default is {0})".format(default_values)
                else:
                    key += sep + "default is {0})".format(default_values)

            rst.definition_list(key, x.description)

    def get_table_label_dict(self, tables, operator_file):
        table_label_dict = {}
        if tables:
            for table in tables:
                label = ""
                table_prop = {}
                if table.label:
                    label = table.label.strip()
                    if table.header:
                        table_prop["header"] = table.header.split(";")
                    if table.width:
                        table_prop["width"] = table.width.strip()
                    if table.caption:
                        table_prop["caption"] = table.caption.strip()
                    if table.row:
                        data = []
                        for r in table.row:
                            cols = r.split(";")
                            data.append(cols)
                        table_prop["data"] = data
                    else:
                        logger.error(
                            "Error processing file %s Table %s without content",
                            operator_file,
                            label,
                        )
                        exit(1)
                else:
                    logger.error(
                        "Error processing file %s Table without label", operator_file
                    )
                    exit(1)
                if label not in table_label_dict:
                    table_label_dict[label] = table_prop
                else:
                    logger.error(
                        'Error processing file %s More than one table having same label "%s"',
                        operator_file,
                        label,
                    )
                    exit(1)
        return table_label_dict

    def get_constraints(self, rst, op_constraints, operator_file):
        _, li = self.convert_list_to_nested_list(
            op_constraints, 0, 0, [], operator_file
        )
        return li

    def add_description(
        self,
        rst,
        description,
        equation_label_dict,
        table_label_dict,
        operator_file,
        category,
    ):
        global SUB_COUNTER
        sub_list = []
        start = 0
        mod_des = ""
        Table_counter = 1
        for part in finditer(
            r"(\\math|\\inlineimage|\\image|\\table)\{[a-zA-Z0-9_,=./ ]*\}", description
        ):
            if "\math" in part.group(0):
                mod_des += description[start : part.start()]
                rst.paragraph(mod_des)
                mod_des = ""
                start = part.end()
                label = part.group(0)[6:-1].strip()
                if label in equation_label_dict:
                    eq = equation_label_dict[label]
                    rst.insert_formula(eq)
                else:
                    logger.error(
                        'Error processing file %s Unknown formula label "%s"',
                        operator_file,
                        label,
                    )
                    exit(1)
            elif "\inlineimage" in part.group(0):
                attr = part.group(0)[13:-1].split(",")
                img_path = join(self.img_folder, attr[0].strip())
                if isfile(img_path):
                    mod_des += description[start : part.start()] + " |sub_{0}| ".format(
                        SUB_COUNTER
                    )
                    width = ""
                    height = ""
                    for i in range(1, len(attr)):
                        if "width" in attr[i]:
                            width = attr[i].split("=")[1].strip()
                        elif "height" in attr[i]:
                            height = attr[i].split("=")[1].strip()
                    if self.doc_compilation_mode != SPHINX_MODE:
                        if width and height:
                            sub_list.append(
                                "\n.. |sub_{0}| image:: diagrams/{1}\n :height: {2}\n :width: {3}\n".format(
                                    SUB_COUNTER, attr[0].strip(), height, width
                                )
                            )
                        elif width:
                            sub_list.append(
                                "\n.. |sub_{0}| image:: diagrams/{1}\n :width: {2}\n".format(
                                    SUB_COUNTER, attr[0].strip(), width
                                )
                            )
                        elif height:
                            sub_list.append(
                                "\n.. |sub_{0}| image:: diagrams/{1}\n :height: {2}\n".format(
                                    SUB_COUNTER, attr[0].strip(), height
                                )
                            )
                        else:
                            sub_list.append(
                                "\n.. |sub_{0}| image:: diagrams/{1}\n".format(
                                    SUB_COUNTER, attr[0].strip()
                                )
                            )
                    else:
                        sub_list.append(
                            "\n.. |sub_{0}| image:: diagrams/{1}/diagrams/{2}\n".format(
                                SUB_COUNTER, self.doc_name, attr[0].strip()
                            )
                        )
                    SUB_COUNTER += 1
                else:
                    logger.error(
                        'Error processing file %s. Image does not exist "%s"',
                        operator_file,
                        attr[0].strip(),
                    )
                    exit(1)
                start = part.end()
            elif "\image" in part.group(0):
                mod_des += description[start : part.start()]
                rst.paragraph(mod_des)
                mod_des = ""
                start = part.end()
                attr = part.group(0)[7:-1].split(",")
                ## temporary WA for inserting equations in sphinx mode
                label = attr[0].strip().split(".")[0].strip()
                if (
                    0
                ):  ##self.doc_compilation_mode == SPHINX_MODE and label in equation_label_dict:
                    eq = equation_label_dict[label]
                    rst.insert_formula(eq)
                else:
                    img_path = join(self.img_folder, attr[0].strip())
                    if isfile(img_path):
                        width = ""
                        height = ""
                        align = "center"
                        for i in range(1, len(attr)):
                            if "width" in attr[i]:
                                width = attr[i].split("=")[1].strip()
                            elif "height" in attr[i]:
                                height = attr[i].split("=")[1].strip()
                            elif "align" in attr[i]:
                                align = attr[i].split("=")[1].strip()
                        if self.doc_compilation_mode != SPHINX_MODE:
                            rst.insert_image(
                                "{}/{}/diagrams/".format(DOC_FOLDER, self.doc_name)
                                + attr[0].strip(),
                                width,
                                height,
                                align,
                            )
                        else:
                            rst.insert_image(
                                "{}/{}/diagrams/".format(DOC_FOLDER, self.doc_name)
                                + self.doc_name
                                + "/diagrams/"
                                + attr[0].strip(),
                                align=align,
                            )
                    else:
                        logger.error(
                            'Error processing file %s. Image does not exist "%s"',
                            operator_file,
                            attr[0].strip(),
                        )
                        exit(1)
            elif "\\table" in part.group(0):
                mod_des += description[start : part.start()]
                rst.paragraph(mod_des)
                mod_des = ""
                start = part.end()
                label = part.group(0)[7:-1].strip()
                if label in table_label_dict:
                    table = table_label_dict[label]
                    caption = (
                        "Table "
                        + str(category)
                        + "."
                        + str(Table_counter)
                        + "  "
                        + table.get("caption")
                        if table.get("caption")
                        else ""
                    )
                    if Table_counter == 1:
                        rst.new_line()
                    Table_counter += 1
                    if self.doc_compilation_mode != SPHINX_MODE:
                        rst.table(
                            len(table["data"]),
                            len(table["data"][0]),
                            table["data"],
                            table.get("header"),
                            table.get("width"),
                            isSummary=False,
                            isDescTable=True,
                            caption=caption,
                        )
                    else:
                        rst.table(
                            len(table["data"]),
                            len(table["data"][0]),
                            table["data"],
                            table.get("header"),
                            table.get("width"),
                            isSummary=False,
                            isDescTable=True,
                            caption=caption,
                            isExt=True,
                        )
                    rst.new_line()
                else:
                    logger.error(
                        'Error processing file %s Unknown formula label "%s"',
                        operator_file,
                        label,
                    )
                    sys.exit(1)
        if not start and not sub_list:
            rst.paragraph(description)
        else:
            rst.paragraph(mod_des + description[start:])
            if sub_list:
                for sub in sub_list:
                    rst.substitution(sub)

    def add_type(self, rst, type_attr):
        for x in type_attr:
            rst.definition_list("{0} : {1}".format(x.name, x.definition), x.description)

    def add_io(self, rst, op_io, h=""):
        for x in op_io:
            if x.optional == True:
                rst.definition_list(
                    "{0} (optional) : {1}".format(x.name, x.type),
                    "{0} (optional)".format(x.description),
                )
            else:
                rst.definition_list("{0} : {1}".format(x.name, x.type), x.description)

    def add_category_introduction(
        self, rst, category, category_num, category_introduction_dict, op_file_dict
    ):
        if category in category_introduction_dict:
            introduction = category_introduction_dict[category]
            op = introduction.op[0]
            equation_label_dict = self.construct_equation_label_dict(
                op.formula, op_file_dict[op.name]
            )
            table_label_dict = self.get_table_label_dict(
                op.table, op_file_dict[op.name]
            )
            if len(op.summary) > 0:
                rst.paragraph(op.summary)
            if len(op.description) > 0:
                self.add_description(
                    rst,
                    op.description,
                    equation_label_dict,
                    table_label_dict,
                    op_file_dict[op.name],
                    category=str(category_num + 2) + ".1",
                )
            if len(op.constraints) > 0:
                rst.paragraph("Restrictions:")
                li = self.get_constraints(rst, op.constraints, op_file_dict[op.name])
                rst.unordered_list(li)

    def add_op_documentation(
        self,
        rst,
        category_op_dict,
        category_introduction_dict,
        kernel_struct_dict,
        op_file_dict,
    ):
        for index, category in enumerate(category_op_dict):
            if self.doc_compilation_mode != SPHINX_MODE:
                rst.page_break()
            rst.h1(category.title())
            self.add_category_introduction(
                rst, category, index, category_introduction_dict, op_file_dict
            )
            self.add_category_summary_table(
                rst, category_op_dict[category], category=[index + 1, category]
            )
            operators = category_op_dict[category]

            if self.doc_compilation_mode != SPHINX_MODE:
                rst.page_break()
            for op_id, operator in enumerate(operators):
                op = operator.op[0]
                equation_label_dict = self.construct_equation_label_dict(
                    op.formula, op_file_dict[op.name]
                )
                table_label_dict = self.get_table_label_dict(
                    op.table, op_file_dict[op.name]
                )
                rst.h2(op.name)
                if len(op.summary) > 0:
                    rst.paragraph(op.summary)
                if len(op.description) > 0:
                    self.add_description(
                        rst,
                        op.description,
                        equation_label_dict,
                        table_label_dict,
                        op_file_dict[op.name],
                        category=str(index + 2) + "." + str(op_id + 1),
                    )
                if op.input:
                    rst.paragraph("Inputs:")
                    self.add_io(rst, op.input)
                if op.output:
                    rst.paragraph("Outputs:")
                    self.add_io(rst, op.output)
                if len(op.c_structure) > 0:
                    rst.paragraph("Params Structure:")
                    rst.insert_code(op.c_structure, "C++")
                elif len(op.structure_name) > 0:
                    n_opname = op.structure_name.strip()
                    if n_opname in kernel_struct_dict:
                        rst.paragraph("Params Structure:")
                        rst.insert_code(kernel_struct_dict[n_opname], "C++")
                if op.attr:
                    rst.paragraph("Attributes:")
                    self.add_attr(rst, op.attr)
                if op.type:
                    rst.paragraph("Types:")
                    self.add_type(rst, op.type)
                if len(op.guid) > 0:
                    data = []
                    for guid_name in op.guid:
                        data.append([guid_name])
                    if self.doc_compilation_mode != SPHINX_MODE:
                        rst.table(len(data[0]), 1, data, ["GUID"], isGUIDtable=True)
                    else:
                        rst.table(
                            len(data[0]),
                            1,
                            data,
                            ["GUID"],
                            isGUIDtable=True,
                            isExt=True,
                        )
                if len(op.constraints) > 0:
                    rst.paragraph("Restrictions:")
                    li = self.get_constraints(
                        rst, op.constraints, op_file_dict[op.name]
                    )
                    rst.unordered_list(li)

    def GenerateRsp(self, additional_appendix=None):
        rst = RstWriter(
            self.doc_compilation_mode == SPHINX_MODE
        )  # Instantiates RstWriter which contains utility functions for inserting text in rst format
        branch_version = self.get_branch_version()
        if self.doc_compilation_mode != SPHINX_MODE:
            rst.header(
                self.doc_name.title()
                + " Synapse Library Reference Rev. "
                + branch_version
            )
            rst.footer(
                [
                    "Habana Labs",
                    "Confidential and Proprietary Information",
                    "###Page###",
                ]
            )
            rst.insert_image(COMPANY_LOGO_PATH, height="3cm")
            rst.title("SynapseAPI " + branch_version)
            rst.subtitle(self.doc_name.title() + " Synapse Library Reference")
            rst.page_break()
            rst.page_counter(template="lowerroman")
            rst.paragraph(COPYRIGHT)
            rst.table_of_contents()
            rst.page_break()
            rst.page_counter()
        else:
            rst.title(self.doc_name.title() + "Synapse Library Reference")

        rst.h1("Introduction")
        category_op_dict, category_introduction_dict, op_file_dict = self.get_all_ops()
        intro_content = self.get_introduction(category_op_dict.keys())
        rst.paragraph(intro_content)
        kernel_struct_dict = self.parse_perf_lib_struct()
        self.add_op_documentation(
            rst,
            category_op_dict,
            category_introduction_dict,
            kernel_struct_dict,
            op_file_dict,
        )
        if self.doc_compilation_mode != SPHINX_MODE:
            rst.page_break()
        # rst.h1("Appendix")
        # self.add_appendix(rst)
        # if (additional_appendix != None):
        #     self.get_appendix(rst, additional_appendix)
        rst.create_rst_file(self.out_file)


if __name__ == "__main__":
    DOC_NAME = argv[1]
    DOC_COMPILATION = argv[2]
    rspgen = OpDef2Rst(DOC_NAME, DOC_COMPILATION)
    # ADDITIONAL_APPENDIX_FILE = argv[3]
    rspgen.GenerateRsp()
