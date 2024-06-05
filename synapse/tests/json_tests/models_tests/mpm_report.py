

import json
import os
import xml.etree.ElementTree as ET
from typing import List
from xml.dom import minidom

from mpm_logger import LOG
from mpm_types import ERROR_CODES, ComObj


def get_tests_name(data):
    names = {}
    for e in data:
        rep = e.report.serialize()
        names[rep.get("name")] = rep.get("timestamp")

    return [k for k, _ in sorted(names.items(), key=lambda item: item[1])]


def header_prettify(header):
    ret = str(header).replace("_", " ").capitalize()
    if "gain" in header:
        ret += " [%]"
    if "time" in str(header):
        ret += " [ms]"
    return ret


def generate_csv_header_line(results):
    ret = ""
    for v in results:
        ret += header_prettify(v) + ","
    return ret + "\n"


def generate_csv_data_line(header_keys, results):
    ret = ""
    for k in header_keys:
        v = results[k] if k in results else ""
        if type(v) == float:
            v = round(v, 3)
        ret += str(v) + ","
    return ret


def generate_csv(data, is_graphs):
    lines = []
    ret = ""
    header_keys = []
    for e in data:
        rep = e.report.serialize()
        if len(rep.keys()) > len(header_keys):
            header_keys = [k for k in rep.keys() if not k.startswith("_")]
    ret = generate_csv_header_line(header_keys)
    for e in data:
        rep = e.report.serialize()
        lines.append(generate_csv_data_line(header_keys, rep))

    for l in lines if is_graphs else sorted(lines):
        ret += l + "\n"
    return ret


def get_result_by_model(data, res_type):
    ret = {}
    for v in res_type:
        ret[v] = {}
    for e in data:
        rep = e.report.serialize()
        for v in res_type:
            if rep.get(v) is None:
                continue
            if rep.get("model") not in ret.get(v):
                ret[v][rep.get("model")] = {}
            ret[v][rep.get("model")][rep.get("name")] = rep.get(v)
    return ret


def get_val_for_result_type(results, type, tests_names):
    lines = []
    result_type = results.get(type)
    for k, v in result_type.items():
        l = f"{k},"
        for t in tests_names:
            l += f"{round(float(v.get(t)),3)}," if v.get(t) is not None else ","
        lines.append(l)
    return lines


def generate_test_compare_csvs(name, data, tests_names, res_type):
    res = get_result_by_model(data, res_type)
    ret = {}
    for rt in res_type:
        lines = get_val_for_result_type(res, rt, tests_names)
        pretty_names = [f"{n} - {header_prettify(rt)}" for n in tests_names]
        csv = f'Model,{",".join(pretty_names)}\n'
        for l in sorted(lines):
            csv += l + "\n"
        ret[f"{name}_{rt}"] = csv
    return ret


def generate_xml(data: List[ComObj]):
    ret = ET.Element("testsuites")
    testsuite = ET.SubElement(ret, "testsuite")
    testsuite.set("name", "models_tests")
    testsuite.set("tests", str(len(data)))
    testsuite.set("errors", str(len([c for c in data if c.status])))
    for com in data:
        if not com.under_test:
            continue
        rep = com.report
        testcase = ET.SubElement(testsuite, "testcase")
        for k, v in rep.get_xml_fields().items():
            testcase.set(k, str(v))
        for m, s in com.status.items():
            if s.status == ERROR_CODES.ERROR:
                error = ET.SubElement(testcase, "error")
                error.set("type", m.name.lower())
                if s.messages:
                    msg = (
                        f"{os.linesep}{os.linesep.join(s.messages)}"
                        f"{os.linesep}repro:{os.linesep}{os.linesep.join(s.repro)}"
                    )
                    error.set("message", msg)
                    error.text = msg
    return str(minidom.parseString(ET.tostring(ret)).toprettyxml(indent="   "))


def export_report(data: List[ComObj], file_type, file_path, is_graphs):
    LOG.info(
        "generate report, type: {}, file path: {}".format(
            file_type, os.path.abspath(file_path)
        )
    )
    if file_type == "json":
        reports = [c.report.serialize() for c in data]
        with open(file_path, "w") as o_file:
            json.dump(reports, o_file, indent=4, sort_keys=True)
        return
    if file_type == "csv":
        csv = generate_csv(data, is_graphs)
        with open(file_path, "w") as o_file:
            o_file.write(csv)
        tests_names = get_tests_name(data)
        if len(tests_names) > 1:
            compare_csvs = generate_test_compare_csvs(
                file_path.replace(".csv", ""),
                data,
                tests_names,
                ["avg_run_time", "compile_time", "run_gain", "compile_gain"],
            )
            for k, v in compare_csvs.items():
                compare_file_path = f"{k}.csv"
                LOG.info(f"compare report: {os.path.abspath(compare_file_path)}")
                with open(compare_file_path, "w") as o_file:
                    o_file.write(v)
        return
    if file_type == "xml":
        xml = generate_xml(data)
        with open(file_path, "w") as o_file:
            o_file.write(xml)
        return
    raise RuntimeError(f"unsupported file file type: {file_type}")


def create_repro_instructions(data: List[ComObj], file_path):
    LOG.info(f"generate reproduction instructions, at: {os.path.abspath(file_path)}")
    repro = "run each of the commands below to reproduce a model failure:\n\n"
    for com in data:
        if not com.under_test:
            continue
        for m, s in com.status.items():
            if s.status == ERROR_CODES.ERROR:
                repro += f"type: {m.name.lower()}, cmd:{os.linesep}{os.linesep.join(s.repro)}\n"
                repro += f"\n"
    with open(file_path, "w") as f:
        f.write(repro)