#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import importlib
import logging
import os
import shutil
import sys
import time
from typing import Dict, List
import yaml
import syn_infra as syn

graph_editor_path = os.path.abspath(
    os.path.join(os.environ.get("SYNAPSE_ROOT"), "tests", "json_tests", "models_tests")
)
sys.path.append(os.path.abspath(graph_editor_path))
editor = importlib.import_module("mpm-editor")

MODELS_REPO = "ssh://gerrit.habana-labs.com:29418/mpm-test-data"

LOG = None

DEVICE_TO_FLAVOR = {
    "gaudi": "g1",
    "gaudi2": "g2",
    "gaudi3": "g1"
}

DEVICE_TO_ACTIONS = {
    "gaudi": "",
    "gaudi2": "",
    "gaudi3": "-a compile"
}

SUPPORTED_DEVICES = {
    "g1": ["gaudi", "gaudi2", "gaudi3"],
    "g2": ["gaudi2", "gaudi3"],
    "g3": ["gaudi3"]
}

DEVICE_DEFAULT_JOBS = {
    "gaudi": ["post-submit", "compilation", "promotion"],
    "gaudi2": ["post-submit", "compilation", "promotion"],
    "gaudi3": ["compilation"]
}

def get_files(folder, postfix) -> List[str]:
    if not folder:
        return
    yield from (
        f
        for f in glob.iglob(f"{folder}/**/*{postfix}", recursive=True)
        if os.path.isfile(f)
    )

class DeviceConfig():
    def __init__(self, chip_type: str, remote_work_folder: str, local_output_folder: str):
        self.chip_type: str = chip_type
        self.local_results_path = os.path.join(local_output_folder, f"{chip_type}")
        self.local_models_file_path = os.path.join(self.local_results_path, "combined_models_list.json")
        self.remote_results_path = os.path.join(remote_work_folder, f"{chip_type}")
        self.local_yaml_path = os.path.abspath(os.path.join(local_output_folder, f"models_tests_{chip_type}.yaml"))

class Config():
    def __init__(self, args):
        self.local_output_folder: str = args.output_folder
        self.local_lfs_folder: str = os.path.join(self.local_output_folder, "mpm")
        self.local_lfs_models_folder: str = os.path.join(self.local_lfs_folder, "models")
        self.local_lfs_models_file_path = os.path.join(self.local_lfs_models_folder, ".default.models-list.json")
        self.remote_work_folder: str = os.path.join("/software", "users", os.getlogin(), "models-update", f"run-{time.strftime('%Y%m%d-%H%M%S')}")
        self.remote_models_path: str = os.path.abspath(os.path.join(self.remote_work_folder, "models"))
        self.remote_bin_path: str = os.path.abspath(os.path.join(self.remote_work_folder, "bin"))
        self.local_qnpu_path: str = os.path.abspath(os.path.join(self.remote_work_folder, "bin"))
        self.jsons: List[str] = args.jsons
        self.jobs: List[str] = args.jobs
        self.use_local_build: bool = args.use_local_build
        self.qa_rec: bool = args.qa_rec
        self.qnpu_version: str = args.qnpu_version
        self.qnpu_build: str = args.qnpu_build
        self.devices: Dict[str, DeviceConfig] = {c:DeviceConfig(c, self.remote_work_folder, self.local_output_folder) for c in args.chip_type}

def config_logger(config: Config):
    global LOG
    LOG = syn.config_logger("models_update", logging.INFO, os.path.join(config.local_output_folder, f"run-{time.strftime('%Y%m%d-%H%M%S')}.log"))
    ch = logging.StreamHandler()
    formatter = logging.Formatter(syn.prepare_logger_format())
    ch.setFormatter(formatter)
    LOG.addHandler(ch)


def create_yaml(deviceConfig: DeviceConfig, config: Config):
    actions = DEVICE_TO_ACTIONS.get(deviceConfig.chip_type)
    container = {
        "name": "models-tests",
        "image": "artifactory-kfs.habana-labs.com/devops-docker-local/habana-builder:ubuntu20.04",
        "command": ["bash", "-c"],
        "args": [
            f"rm -rf {deviceConfig.remote_results_path} && cp -Lr {config.remote_bin_path} $WORKSPACE/builds && cp -r {config.remote_models_path} $WORKSPACE/models && source $WORKSPACE/repos/jenkins_files/src/env/habana_ci_env $WORKSPACE $WORKSPACE/repos $WORKSPACE/builds && git -C $WORKSPACE/repos/mpm-test-data lfs pull && run_models_tests -l d -c {deviceConfig.chip_type} -w {deviceConfig.remote_results_path} --models_folder $WORKSPACE/models/{deviceConfig.chip_type} {actions}"
        ],
        "env": [{"name": "WORKSPACE", "value": "/root"}],
        "workingDir": "/root",
    }
    yaml_content = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "TFJob",
        "metadata": {
            "name": "Models-Tests",
            "annotations": {
                "habana.ai/automation": "master",
                "habana.ai/hl_driver_version": "master",
                "habana.ai/nic_kmd": "master",
                "pod-reaper/max-duration": "450m",
                "repos": "jenkins_files:master,automation:master,synapse:master,mpm-test-data:master_next",
            },
        },
        "spec": {
            "tfReplicaSpecs": {
                "Worker": {
                    "replicas": 1,
                    "template": {
                        "spec": {
                            "containers": [container],
                            "hostIPC": True,
                            "hostPID": True,
                        }
                    },
                }
            }
        },
    }
    with open(deviceConfig.local_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)


def create_manifest(qnpu_repo: str) -> str:
    json_test_manifest = []
    json_test_manifest_file_path = os.path.join(
        qnpu_repo, "qnpu-manifests", "json_test.xml"
    )
    json_test_manifest.append('<?xml version="1.0" encoding="UTF-8"?>')
    json_test_manifest.append("<manifest>")
    json_test_manifest.append(
        '<remote  name="gerrit" alias="origin" fetch="ssh://gerrit.habana-labs.com:29418" review="gerrit"/>'
    )
    json_test_manifest.append(
        '<default revision="master" remote="gerrit" sync-j="4" />'
    )
    json_test_manifest.append('<project name="automation"/>')
    json_test_manifest.append('<project name="synapse"/>')
    json_test_manifest.append("</manifest>")
    with open(json_test_manifest_file_path, "w", encoding="utf-8") as f:
        f.writelines(json_test_manifest)
    return json_test_manifest_file_path


def generate_bin_folder_from_qnpu(config: Config):
    qnpu_repo = os.path.join(os.path.expanduser("~"), ".qnpu")
    env = {}
    env["PATH"] = f'{qnpu_repo}:{os.environ.get("PATH")}'
    venv_file_path = os.path.join(os.path.expanduser("~"), "jtVenv")
    if not os.path.exists(venv_file_path):
        syn.run_external(
            ["python3", "-m", "pip", "install", "virtualenv", "--user"], env, None, True
        )
        syn.run_external(["virtualenv", "-p", "python3", "venv_file_path"], env, None, True)
        syn.git_clone(
            "ssh://gerrit.habana-labs.com:29418/automation",
            os.path.join("/tmp", "automation_repo"),
        )
        requirements = os.path.join(
            "/tmp", "automation_repo", "ci", "requirements-training.txt"
        )
        syn.run_external(["pip", "install", "-r", requirements], env, None, True)

    if os.path.exists(qnpu_repo):
        syn.git_pull(qnpu_repo, None, LOG)
    else:
        syn.git_clone("ssh://gerrit.habana-labs.com:29418/devops/qnpu", qnpu_repo)

    json_test_manifest_file_path = create_manifest(qnpu_repo)
    qnpu_name = "models-update"
    local_qnpu_path = os.path.join(os.path.expanduser("~"), "qnpu", qnpu_name)
    if os.path.exists(local_qnpu_path):
        shutil.rmtree(local_qnpu_path)
    qnpu_cmd = [os.path.join(qnpu_repo, "qnpu-init.sh"), "--os", "ubuntu2004", "-m", json_test_manifest_file_path, qnpu_name]
    if config.qnpu_version:
        qnpu_cmd.extend(["-v", config.qnpu_version])
    if config.qnpu_build:
        qnpu_cmd.extend(["-b", config.qnpu_build])
    syn.run_external(qnpu_cmd, env, LOG, True)
    LOG.info('copy bin folder from: %s to: %s', os.path.join(local_qnpu_path, "bin"), config.remote_bin_path)
    shutil.copytree(os.path.join(local_qnpu_path, "bin"), config.remote_bin_path, symlinks=False)
    shutil.rmtree(local_qnpu_path)


def generate_bin_folder_from_local_build(remote_bin_path: str):
    LOG.info('copy bin folder from: %s to: %s', os.environ.get("BUILD_ROOT"), remote_bin_path)
    shutil.copytree(os.environ.get("BUILD_ROOT"), remote_bin_path, symlinks=False)


def generate_bin_folder(config: Config):
    LOG.info("generate bin folder at: %s", config.remote_bin_path)
    if os.path.exists(config.remote_bin_path):
        shutil.rmtree(config.remote_bin_path)
    if config.use_local_build:
        generate_bin_folder_from_local_build(config.remote_bin_path)
    else:
        generate_bin_folder_from_qnpu(config)


def generate_models_folder(config: Config):
    LOG.info("generate models folder at: %s", config.remote_models_path)
    if os.path.exists(config.remote_models_path):
        shutil.rmtree(config.remote_models_path)
    os.makedirs(config.remote_models_path)
    if config.qa_rec:
        base_folder = "/qa/synrec_dumps/latest"
        folders = [os.path.join(base_folder, "ind", "pre_graphs"), os.path.join(base_folder, "pol", "pre_graphs"), os.path.join(base_folder, "isr", "graphs")]
        for folder in folders:
            if os.path.exists(folder):
                jsons = get_files(folder, ".json")
                for j in jsons:
                    model_name = os.path.basename(j).replace(".json", "")
                    if model_name.startswith("tf_"):
                        LOG.warning("TF models are not allowed in the model name, model: %s", model_name)
                        continue
                    device_short_name = model_name[len(model_name) - 2:]
                    if device_short_name not in SUPPORTED_DEVICES:
                        LOG.warning("Invalid model name: %s, model name should end with: %s", model_name, " or ".join(SUPPORTED_DEVICES.keys()))
                        continue
                    devices = SUPPORTED_DEVICES[device_short_name]
                    for d in devices:
                        os.makedirs(os.path.join(config.remote_models_path, d), exist_ok=True)
                        shutil.copy(j, f"{os.path.join(config.remote_models_path, d)}/")
    else:
        for j in config.jsons:
            if os.path.isdir(j):
                shutil.copytree(j, config.remote_models_path, symlinks=False, dirs_exist_ok=True)
            else:
                shutil.copy(j, config.remote_models_path)


def launch_device(dev_conf: DeviceConfig, config: Config):
    try:
        jobs = config.jobs if config.jobs else dev_conf.chip_type
        create_yaml(dev_conf, config)
        flavor = DEVICE_TO_FLAVOR.get(dev_conf.chip_type)
        container_cmd = ["hlctl", "create", "containers", "-f", dev_conf.local_yaml_path, "--flavor", flavor, "--watch", "--mount-sshkeys"]
        syn.run_external(container_cmd, None, LOG, True)
        shutil.copytree(dev_conf.remote_results_path, dev_conf.local_results_path)
        editor.add(dev_conf.local_models_file_path, dev_conf.local_models_file_path, None, [dev_conf.chip_type], jobs, None)
        editor.update(config.local_lfs_models_file_path, config.local_lfs_models_file_path, dev_conf.local_models_file_path)
    except Exception as e:
        LOG.error("launch_device failed, error: %s", e)
        sys.exit(-1)


def launch(config: Config):
    LOG.info("launch models tests on: %s", " ".join(config.devices.keys()))
    tpe = ThreadPoolExecutor(len(config.devices))
    for dev_conf in config.devices.values():
        tpe.submit(launch_device, dev_conf, config)
    tpe.shutdown()


def create_models_folders(config: Config) -> str:
    LOG.info("create models folder at: %s", config.local_lfs_models_folder)
    try:
        syn.git_clone(MODELS_REPO, config.local_lfs_folder, {}, LOG)
        syn.git_checkout(config.local_lfs_folder, "master_next", None, LOG)
        syn.git_pull(config.local_lfs_folder, None, LOG)
        jsons = get_files(config.remote_models_path, ".json")
        for j in jsons:
            shutil.copy(j, f"{config.local_lfs_models_folder}/")
    except Exception as e:
        LOG.error("create_models_folders failed, error: %s", e)
        sys.exit(-1)


def collect_models_stats(config: Config):
    try:
        generate_models_folder(config)
        generate_bin_folder(config)
        launch(config)
    except Exception as e:
        LOG.error("collect_models_stats failed, error: %s", e)
        sys.exit(-1)


def main(args):
    config = Config(args)
    if os.path.exists(config.local_output_folder):
        if not args.overwrite:
            raise RuntimeError(f"output folder exists, use --overwrite to overwrite it, path: {config.local_output_folder}")
        shutil.rmtree(config.local_output_folder)
    os.makedirs(config.local_output_folder)
    config_logger(config)
    LOG.info("command: %s", " ".join(sys.argv))
    LOG.info('vscode args: "args": ["%s"]', '", "'.join(sys.argv[1:]))
    tpe = ThreadPoolExecutor(2)
    tpe.submit(collect_models_stats, config)
    tpe.submit(create_models_folders, config)
    tpe.shutdown()
    syn.run_external(["git", "-C", config.local_lfs_folder, "add", "."], None, LOG)
    commit_message = '[SW-69192] periodic auto models update'
    syn.run_external(["git", "-C", config.local_lfs_folder, "commit", "-m", f'"{commit_message}"'], None, LOG)
    print(
        f"Local commit with the new models is available at: {os.path.abspath(config.local_lfs_folder)}"
    )
    print("To update the the remote models:")
    print(
        f"cd {os.path.abspath(config.local_lfs_folder)} && git push origin HEAD:refs/for/master_next"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    default_devices = ["gaudi", "gaudi2", "gaudi3"]

    parser.add_argument(
        "-o",
        "--output-folder",
        help="Output folder path",
        default=f"models-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "-c",
        "--chip-type",
        choices=list(syn.CHIP_TYPES.keys()),
        nargs="+",
        help=f"Select chip type (default: {' '.join(default_devices)})",
        default=default_devices,
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["promotion"],
        help="Set job filter for the new models",
    )
    parser.add_argument(
        "--qnpu-version",
        help="Use specific qnpu version",
    )
    parser.add_argument(
        "--qnpu-build",
        help="Use specific qnpu build",
    )
    parser.add_argument(
        "--use-local-build",
        action="store_true",
        help="Use local builds folder insted pulling qnpu binaries"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output folder"
    )
    jsons_parser = parser.add_mutually_exclusive_group()
    jsons_parser.add_argument(
        "-j",
        "--jsons",
        nargs="+",
        help="List of config to file/folder of jsons",
    )
    jsons_parser.add_argument(
        "--qa-rec",
        action="store_true",
        help="Update jsons from QA recordings"
    )

    ret = parser.parse_args()
    if ret.qnpu_build and not ret.qnpu_version:
        parser.error("trying to select qnpu build without specify a version")
    return ret


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
