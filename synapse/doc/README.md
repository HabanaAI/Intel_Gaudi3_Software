# The Synapse Documentation Infrastructure

This directory and its subdirectories contains pbtxt files for various operators supported on different devices and source code for parsing 
these pbtxt files into restructured text.

The README briefly describes how to get started with writing documentation files and modifying the source code for parsing.

## Directory Structure

doc
|__ common - contains proto structure, source code for parsing pbtxt content, stylesheet and other image files which are common across all devices
    |__ protobuf - contains proto structure
        |__ op_def.proto - proto structure which specifies data fields and their type for writing pbtxt file. If any new information needs to be passed to rst then the corresponding data field needs to be added in op_def.proto. This file is used by the python script to parse the contents of .pbtxt files.
    |__ complexopdef2rst.py - python script for generating documentation of complex ops. Internally calls opdef2rst to achieve the same.
    |__ opdef2rst.py - python script for generating restructured text from the .pbtxt files. All the logic concerning how the parsed data will be rendered resides in this file. For example, the parsed data can be displayed as a list or as a table, now the decision whether it is displayed as a list or as a table is specified in this file. So any change in placement of the data in final documentation will require a change in this file.
    |__ rstwriter.py - python script which provides the infrastructure for inserting the content in restructered text format. It is used by opdef2rst.py.
    |__ style.style - stylesheet specifying all the properties related to look and feel of the documentation pdf. For example, font, font size, font color, background color etc. It is used by rst2pdf while converting restructured text into pdf format.
    |__ ...
|__ gaudi - contains documentation data related to gaudi device
    |__ appendix - contains appendix files
        |__ appendix1.txt
        |__ appendix2.txt
        |__ ...
    |__ complex_op_appendix - xontains appendix files for complex operators
        |__ appendix1.txt
        |__ ...
    |__ complex_op_def - contains pbtxt files for complex operators
        |__ category_a
            |__ introduction.pbtxt
            |__ op1.pbtxt
            |__ op2.pbtxt
            |__ ...
        |__ category_b
            |__ ...
        |__ ...
    |__ op_def - contains pbtxt files for gaudi operators. Sub directory structure same as complex_op_def
        |__ ...
    |__ diagrams - contains image files referred in pbtxt files. All the images which are referred in pbtxt file should be present here otherwise error will be raised while parsing pbtxt file.
        |__ fig1.png
        |__ ...
|__ gaudi2 - contains documentation data related to gaudi2 device. sub directory structure same as gaudi.
    |__ ...
|__ goya1 - contains documentation data related to goya1 device. sub directory structure same as gaudi.
    |__ ...
|__ ...

## Description of proto structure defined in op_def.proto

*constraints* : repeated string
    A list of string used to define restrictions imposed by the operator. List should begin with a opening square bracket `[` and end with a closing square bracket `]`. Since it is a list of string so contraint should be enclosed within double quotes. If there are multiple constraints then they should be separated by a comma. If a constraint needs to be inserted as a sub-level of previous constraint then it should begin with a dollar `$` symbol. The number of dollar symbols determines the level of nesting. _Example: constraints:["C0", "$C00", "$C01", "$$C010", "C1"]_ will be rendered as:
    - C0
      - C00
      - C01
        - C010
    - C1

## Note

The data in .pbtxt files is first parsed by a python script which converts it into rst format. Therefore, any substring in .pbtxt which contains escape sequences recognized in python and needs to inserted as is in rst format, should have an extra backslash character. _Example: \begin, \text, \ , \frac should be wriiten as \\begin, \\text, \\ , \\frac respectively._