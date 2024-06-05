# SPP - Habana Labs Synapse Performance Primitives library

## Requirements

To build SPP, you need the following libraries installed on your computer:

- cmake
- doxygen (optional)

In Ubuntu, run the following commands:

$ sudo apt install git g++ cmake doxygen

## Build debug

To build the project, run
    $ mkdir debug
    $ cd debug
    $ cmake -DDALI=ON ..
    $ make


## build release
    $ mkdir release
    $ cd release
    $ cmake -DDALI=ON -DCMAKE_BUILD_TYPE=Release ..
    $ make

## Doxygen

Documentation can be generated with

    $ make doc

A tutorial on the doxygen syntax can be found [here](http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html).

## googletest

You can also make use of the googletest library to write and execute automated tests.
Simply create new tests in the `tests` directory.
They will be automatically compiled into an executable and can be run with

    make check

For an introduction to writing tests with googletest, follow [this link](http://code.google.com/p/googletest/wiki/Primer).

