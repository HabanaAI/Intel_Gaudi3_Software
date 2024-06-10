# Intel&reg; Gaudi%reg; Software Open Source Project

Intel Gaudi Software for Open Source

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Welcome to the Intel Gaudi Software Open Source Project 

The Intel Gaudi Software Open Source Project is shared by the Intel Gaudi team with the broader AI community. By providing this project to the open-source community, The Intel Gaudi team encourages the development of cutting-edge AI technologies and solutions that benefit researchers, developers, and enthusiasts worldwide.

## Installation

### Step 1: Deployment

1. **Download Package:** Obtain the package containing all project files from [source location].

2. **Extract Files:** Extract the contents of the package to your desired location on your system.

### Step 2: Environment Setup

1. **Run habana_set_env_val.sh:**
   - Open a terminal window.
   - Navigate to the package directory.
   - Execute the following command: 
     ```bash
     source trees/habana_set_env_val.sh ./trees ./builds
     ```

### Step 4: Building Projects

1. **Build hl_logger:**
   - After setting up the environment, run the following command:
     ```bash
     build_hl_logger [-r]
     ```
   This command will build the `hl_logger` project. Ensure that it is built successfully before proceeding.

2. **Build hl_gcfg:**
   - To build the `hl_gcfg` project, use the following command:
     ```bash
     build_hl_gcfg [-r]
     ```
3. **Build scal:**
   - To build the `scal` project, use the following command:
     ```bash
     build_scal [-r]
     ```

4. **Build mme:**
   - To build the `mme` project, use the following command:
     ```bash
     build_mme [-r]
     ```
4. **Build synapse:**
   - To build the `synapse` project, use the following command:
     ```bash
     build_synapse [-r]
     ```


**Note:** It's essential to build with this order.

**Note:** Before running the tests, if the engines_fw or rdma-core package is not installed,
  ensure that the following environment variables point to the correct locations:

- For **eng_XXX.bin**: export HABANA_SCAL_BIN_PATH=PATH/IN/YOUR/SYSTEM/builds/engines_fw_release_build
- For **libhlib.so**: export RDMA_CORE_LIB=PATH/IN/YOUR/SYSTEM/trees/npu-stack/rdma-core/build/lib


## License

Information about the project's license. Specify the license type and provide a link to the full license text if applicable.
