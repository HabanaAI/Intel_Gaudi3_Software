#!/bin/bash
#
# Copyright (C) 2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Author: Shai Kedem <skedem@habana.ai>
#

#
# You can run this script from any dir
#

function install_iwyu()
(
    # Colors
    function _set_colors()
    {
        local -r colors="local -r Black='\033[0;30m' ; local -r Red='\033[0;31m' ; local -r Green='\033[0;32m'; local -r Yellow='\033[0;33m'; local -r Blue='\033[0;34m'; local -r Purple='\033[0;35m'; local -r Cyan='\033[0;36m'; local -r White='\033[0;37m'; local -r BBlack='\033[1;30m'; local -r BRed='\033[1;31m'; local -r BGreen='\033[1;32m'; local -r BYellow='\033[1;33m'; local -r BBlue='\033[1;34m'; local -r BPurple='\033[1;35m'; local -r BCyan='\033[1;36m'; local -r BWhite='\033[1;37m'; local -r UBlack='\033[4;30m'; local -r URed='\033[4;31m'; local -r UGreen='\033[4;32m'; local -r UYellow='\033[4;33m'; local -r UBlue='\033[4;34m'; local -r UPurple='\033[4;35m'; local -r UCyan='\033[4;36m'; local -r UWhite='\033[4;37m'; local -r On_Black='\033[40m'; local -r On_Red='\033[41m'; local -r On_Green='\033[42m'; local -r On_Yellow='\033[43m'; local -r On_Blue='\033[44m'; local -r On_Purple='\033[45m'; local -r On_Cyan='\033[46m'; local -r On_White='\033[47m'; local -r IBlack='\033[0;90m'; local -r IRed='\033[0;91m'; local -r IGreen='\033[0;92m'; local -r IYellow='\033[0;93m'; local -r IBlue='\033[0;94m'; local -r IPurple='\033[0;95m'; local -r ICyan='\033[0;96m'; local -r IWhite='\033[0;97m'; local -r BIBlack='\033[1;90m'; local -r BIRed='\033[1;91m'; local -r BIGreen='\033[1;92m'; local -r BIYellow='\033[1;93m'; local -r BIBlue='\033[1;94m'; local -r BIPurple='\033[1;95m'; local -r BICyan='\033[1;96m'; local -r BIWhite='\033[1;97m'; local -r On_IBlack='\033[0;100m'; local -r On_IRed='\033[0;101m'; local -r On_IGreen='\033[0;102m'; local -r On_IYellow='\033[0;103m'; local -r On_IBlue='\033[0;104m'; local -r On_IPurple='\033[0;105m'; local -r On_ICyan='\033[0;106m'; local -r On_IWhite='\033[0;107m'; local -r NC='\033[0m'"
        echo "${colors}"
    }

    function _print_error()
    {
        eval $( _set_colors )
        local -r str="$1"
        echo -e "${BIRed}${str}${NC}"
    }

    function _print_info_ok()
    {
        eval $( _set_colors )
        local -r str="$1"
        echo -e "${IGreen}${str}${NC}"
    }

    #
    # install homebrew, simulate enter keypress
    echo -ne '\n' | /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || { _print_error "Unable to install homebrew"; return 1;  }

    # set path vars
    export HOMEBREW_PREFIX="/home/linuxbrew/.linuxbrew"
    export HOMEBREW_CELLAR="${HOMEBREW_PREFIX}/Cellar"
    export HOMEBREW_REPOSITORY="${HOMEBREW_PREFIX}/Homebrew"

    local -r temp_paths="export MANPATH=\"${HOMEBREW_PREFIX}/share/man:${MANPATH+:$MANPATH}:\"; export INFOPATH=\"${HOMEBREW_PREFIX}/share/info:${INFOPATH:-}\"; export PATH=\"${HOMEBREW_PREFIX}/bin:${HOMEBREW_PREFIX}/sbin:${PATH+:$PATH}\""

    # check brew working
    (  eval "${temp_paths}" && brew --version ) || { _print_error "homebrew was not installed correctly"; return 1; }

    # Install latest version from IWYU
    ( eval "${temp_paths}" && brew install iwyu ) || { _print_error "Unable to install IWYU with homebrew" ; return 1; }

    # check IWYU is working from path
    ( eval "${temp_paths}" && include-what-you-use --version ) || { _print_error "IWYU was not installed correctly"; return 1; }

    _print_info_ok "Successfully installed IWYU"
)
