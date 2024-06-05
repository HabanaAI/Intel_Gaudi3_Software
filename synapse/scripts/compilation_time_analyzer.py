import logging
import syn_infra as syn
import re
import argparse
import csv
import os

LOG = syn.config_logger("compilation_time_analysis", logging.INFO)
LOG_FILES_PREFIX = "synapse_log"
LOG_FILES_POSTFIX = ".txt"
COL_RECIPE_NAME = "Recipe Name"
COL_TESTED_AREA = "Tested Area"
COL_TIME = "Time (sec)"
COL_TIME_FROM_TOTAL = "Total Compile Time [%]"
PASS_TIME_TABLE_HEADER = [
    COL_RECIPE_NAME,
    COL_TESTED_AREA,
    COL_TIME,
    COL_TIME_FROM_TOTAL,
]

num_of_digits_after_point = 3


class Pass:
    def __init__(self, name: str, time: float):
        self.name = name
        self.time = time
        self.time_from_total_passes = 0


class PassContainer:
    def __init__(self):
        self.container: dict[str, Pass] = {}
        self.total_passes_time: float = 0.0

    def add_pass(self, p: Pass):
        if p.name not in self.container:
            self.container[p.name] = p
        else:
            self.container[p.name].time += p.time
        self.total_passes_time += p.time

    def get_passes(self):
        return self.container.values()

    def calculate_time_from_total_for_each_pass(self):
        for p in self.get_passes():
            p.time_from_total_passes = round(
                (p.time / self.total_passes_time) * 100, num_of_digits_after_point
            )


class Recipe:
    def __init__(self, name: str):
        self.name = name
        self.passes = PassContainer()

    def to_list(self):
        l = []
        for p in self.passes.get_passes():
            l.append([self.name, p.name, p.time, p.time_from_total_passes])
        return l


def save_list_to_csv_file(rows, file_path):
    with open(file_path, "w") as o_file:
        writer = csv.writer(o_file)
        for row in rows:
            writer.writerow(row)


class RecipeContainer:
    def __init__(self):
        self.container: list[Recipe] = []

    def get_recipes(self):
        return self.container

    def add_recipe(self, r: Recipe):
        self.container.append(r)

    def add_recipes(self, recipeContainer):
        self.container += recipeContainer.get_recipes()

    def get_last_recipe(self):
        if len(self.container) > 0:
            return self.container[-1]
        return None

    def save_to_csv(self, file_path="passes_times.csv"):
        LOG.info(f"save data to {file_path}")
        recipes_as_list = []
        for recipe in self.get_recipes():
            recipes_as_list += recipe.to_list()
        save_list_to_csv_file([PASS_TIME_TABLE_HEADER] + recipes_as_list, file_path)
        return file_path


class LogParser:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.recipes = RecipeContainer()
        return self.parse_log_to_graph_object()

    def handle_new_recipe(self, pattern):
        name, = pattern.groups()
        self.recipes.add_recipe(Recipe(name))

    def handle_pass(self, pattern):
        pass_name, measured_time = pattern.groups()
        self.recipes.get_last_recipe().passes.add_pass(
            Pass(pass_name, float(measured_time))
        )

    def parse_log_to_graph_object(self):
        LOG.info(f"start parsing {self.log_file_path}")
        pass_manager_scheme_pattern = re.compile(r".*\[PASS_MANAGER.*\s*\] (.*)")
        recipe_name_pattern = re.compile(r".* Recipe name: (.*)")
        pass_name_and_time_pattern = re.compile(r".* Total time for (.*): (.*) seconds")
        with open(self.log_file_path) as infile:
            for line_count, line in enumerate(infile):
                try:
                    if pass_manager_scheme_pattern.search(line):
                        res = pass_name_and_time_pattern.search(line)
                        if res:
                            self.handle_pass(res)
                            continue
                        res = recipe_name_pattern.search(line)
                        if res:
                            self.handle_new_recipe(res)
                            continue
                except Exception as e:
                    LOG.error(
                        "problem occrued in parsing of log file, in line {}".format(
                            line_count
                        )
                    )
                    LOG.error(line)
                    raise e
        LOG.info(f"parsing finished")


def main(args):
    recipes = RecipeContainer()
    for log_file in syn.get_files(
        args.logs_folder, LOG_FILES_POSTFIX, LOG_FILES_PREFIX
    ):
        parser = LogParser(log_file)
        recipes.add_recipes(parser.recipes)
    for recipe in recipes.get_recipes():
        recipe.passes.calculate_time_from_total_for_each_pass()
    return recipes.save_to_csv()


def parse_args(sys_args=None):
    parser = argparse.ArgumentParser(
        add_help=True,
        description="This tool analyze synapse_log files to determine each pass working time and export it to csv.\nIf you encounter any error, please open a ticket and assign to tmagdaci.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    files_test_group = parser.add_mutually_exclusive_group()
    files_test_group.add_argument(
        "-f",
        "--logs-folder",
        help="Path to the folder that contains synapse_log.txt files",
        default=os.getenv("HABANA_LOGS"),
    )
    args, _ = parser.parse_known_args(sys_args)
    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
