print_usage() {
  echo "Missing format option:"
  echo "  Run: './format.sh apply' if you want to apply format."
  echo "  Run: './format.sh check' if you want to check format."
}

fix_format() {
	find ~/trees/npu-stack/media/ -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
}

check_format() {
  touch pre.status
  touch post.status

  git status > pre.status
  fix_format
  git status > post.status

  if ! diff pre.status post.status; then
    echo "ERROR: files need to be formatted"
    echo "***pre.status***"
    cat pre.status
    echo "***post.status***"
    cat post.status
    echo "***git diff***"
    git diff

    exit 1
  fi
}

if [ $# -eq 0 ]
  then
    echo "No arguments supplied, applying format"
    fix_format
    echo "done."
else
  if [ "$1" = "apply" ]; then
    echo "Running apply format."
    fix_format
  elif [ "$1" = "check" ]; then
    echo "Running check format."
    check_format
    echo "done."
  else
    print_usage
    exit 1
  fi
fi
