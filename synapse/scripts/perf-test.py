#!/usr/bin/env python
import argparse
import errno
import getpass
import glob
import json
import logging
import os
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Tuple

import syn_infra as syn
import yaml

MODELS_REPO = "ssh://gerrit:29418/mpm-test-data"
QNPU_REPO = "ssh://gerrit.habana-labs.com:29418/devops/qnpu"
MODELS_REVISION = "master_next"
LOCAL_MODELS_FOLDER = f"/tmp/.mpm"
ARTIFACTORY_PATH = (
    "artifactory-kfs.habana-labs.com/developers-docker-dev-local/gc/users"
)

REGRESSION_THRESHOLD = 2

LOG = logging.getLogger("perf-test")


def config_logger(file_path):
    LOG_LEVEL = logging.DEBUG
    foramt = "%(asctime)s (%(threadName)s) [%(levelname)s]: %(message)s"
    logging.basicConfig(
        filename=file_path,
        format=foramt,
        level=LOG_LEVEL,
    )
    LOG.setLevel(LOG_LEVEL)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(foramt)
    ch.setFormatter(formatter)
    LOG.addHandler(ch)


class DeviceInfo:
    def __init__(self, name: str, container: str):
        self.name: str = name
        self.compile_container: str = "cpu"
        self.run_container: str = container


class DeviceType(Enum):
    T_1 = 1
    T_2 = 2
    T_3 = 3


DEVICES = {
    DeviceType.T_1: DeviceInfo("gaudi", "g1"),
    DeviceType.T_2: DeviceInfo("gaudi2", "g2"),
    DeviceType.T_3: DeviceInfo("gaudi3", "cpu"),
}

NAME_TO_DEVICE_TYPE = {
    "gaudi": DeviceType.T_1,
    "gaudi2": DeviceType.T_2,
}


SUPPORTED_DEVICE: List[DeviceInfo] = [
    DEVICES.get(DeviceType.T_1),
    DEVICES.get(DeviceType.T_2),
]


class QnpuType(Enum):
    MIN = 1
    SYNAPSE = 2


QNPU_MANIFESTS = {
    QnpuType.MIN: "min",
    QnpuType.SYNAPSE: "synapse",
}


class ContainerConfig:
    def __init__(self, yaml_path: str, flavor: str):
        self.yaml_path: str = yaml_path
        self.flavor: str = flavor
        self.status: int = 0


class RunConfig:
    def __init__(self, args):
        self.work_folder: str = args.work_folder
        self.results_folder: str = args.results_folder
        self.docker_folder: str = os.path.join(args.work_folder, "docker")
        self.base_patches: List[str] = args.base_patches
        self.test_patchs: List[str] = args.test_patches
        self.devices: List[str] = get_device_list(args.chip_types)
        self.jobs: List[str] = args.jobs
        self.models_folder: bool = args.models_folder
        self.test_models: bool = args.test_models
        self.models_revision: bool = args.models_revision
        self.checkout: bool = args.checkout
        self.run_once: bool = args.run_once
        self.threads: int = args.threads
        self.image_name: str = (
            f'{ARTIFACTORY_PATH}/{getpass.getuser()}:{time.strftime("%Y%m%d-%H%M%S")}'
        )
        self.stack: QnpuStack = (
            QnpuStackGerrit(
                args.work_folder,
                QnpuType.SYNAPSE,
                args.base_rev,
                args.qnpu_version,
            )
            if args.gerrit
            else QnpuStackGitRevs(
                args.work_folder,
                QnpuType.SYNAPSE,
                args.base_rev,
                args.remote,
                args.qnpu_version,
            )
        )


class GerritInfo:
    def __init__(self, id: str, patch: int):
        self.id: str = id
        self.patch: int = patch


def raise_if(condition: bool, message: str):
    if condition:
        raise RuntimeError(message)


class QnpuStack:
    def __init__(
        self,
        work_folder_abs_path: str,
        qnpu_type: QnpuType,
        base_revision: str,
        qnpu_version: str,
    ):
        self.org_hash = None
        self.type: QnpuType = qnpu_type
        self.work_folder: str = work_folder_abs_path
        self.base_revision: str = base_revision
        self.version: str = None
        self.folder: str = None
        self.activate_file_path: str = None
        self.init(qnpu_version)

    def init(self, qnpu_version):
        LOG.info(f"creating QNPU at: {self.work_folder}")
        version_folder = "latest"
        self.folder: str = os.path.join(self.work_folder, "qnpu", version_folder)
        new_work_folder = not os.path.exists(self.folder)
        version_args = ["-v", qnpu_version]
        if new_work_folder:
            cmd_args = [
                "qnpu-init.sh",
                "-m",
                f"{QNPU_MANIFESTS.get(self.type)}.xml",
                version_folder,
            ]
            if qnpu_version:
                cmd_args += version_args
            sts = syn.run_external(cmd_args, {"HOME": self.work_folder}, LOG, True)
            if sts != 0 and os.path.exists(self.work_folder):
                shutil.rmtree(self.work_folder)
                raise RuntimeError(
                    f"failed to init QNPU at: {os.path.join(self.work_folder, version_folder)}"
                )
        bin_file_log: str = os.path.join(
            self.folder, "rel", ".qnpu-download-binaries.log"
        )
        with open(bin_file_log) as f:
            line = f.readline().split(" ")
            self.version = f"{line[2]}-{line[4]}"
        self.activate_file_path: str = os.path.join(self.folder, "activate")
        if not new_work_folder:
            sync_cmd = (
                f"qnpu-sync.sh {' '.join(version_args)}"
                if qnpu_version
                else "qnpu-sync.sh"
            )
            cmd_args = [
                f'bash -ic "source {self.activate_file_path} && {sync_cmd} && qnpu-deactivate"'
            ]

            syn.run_external(cmd_args, {"HOME": self.work_folder}, LOG, True)

    def get_version(self) -> str:
        return self.version

    def get_src_folder(self) -> str:
        return os.path.join(self.folder, "src")

    def get_bin_folder(self) -> str:
        return os.path.join(self.folder, "bin")

    def apply_patches(self, patchs: List[str], checkout: bool) -> None:
        pass

    def build(self, name: str) -> None:
        cmd_args = [
            f'bash -ic "source {self.activate_file_path} && build_mme -r && build_synapse -r && qnpu-deactivate"',
        ]
        syn.run_external(cmd_args, {"HOME": self.work_folder}, LOG, True)
        lib_folder_path = os.path.join(
            self.get_bin_folder(), "synapse_release_build", "lib"
        )
        shutil.copy2(
            os.path.join(lib_folder_path, "libSynapse.so"),
            os.path.join(lib_folder_path, f"libSynapse.{name}.so"),
            follow_symlinks=True,
        )

    def revert(self) -> None:
        synapse = os.path.join(self.get_src_folder(), "synapse")
        git_cmd = f"git -C {synapse} checkout {self.org_hash}"
        sts = syn.run_external([git_cmd], None, LOG, True)
        if sts != 0:
            raise RuntimeError(f"failed to checkout org commit: {self.org_hash}")
        LOG.info(f"checked out org commit, hash: {self.org_hash}")

    def apply_patches_and_build(
        self, name: str, patchs: List[str], checkout: bool
    ) -> None:
        self.apply_patches(patchs, checkout)
        self.build(name)
        self.revert()


class QnpuStackGerrit(QnpuStack):
    def __init__(
        self,
        work_folder_abs_path: str,
        qnpu_type: QnpuType,
        base_revision: str,
        qnpu_version: str,
    ):
        super().__init__(
            work_folder_abs_path,
            qnpu_type,
            base_revision,
            qnpu_version,
        )

    def get_gerrits_infos(self, gerrits_ids: List[str]) -> List[GerritInfo]:
        ret = []
        if not gerrits_ids:
            return ret
        synapse = os.path.join(self.get_src_folder(), "synapse")
        for id in gerrits_ids:
            ref = f"refs/changes/{id[-2:]}/{id}"
            sts: int = 0
            patch: int = 0
            while sts == 0:
                check_rev = f"git -C {synapse} fetch origin {ref}/{str(patch + 1)}"
                sts = syn.run_external([check_rev], None, LOG, True)
                patch += 1 if sts == 0 else 0
            raise_if(patch == 0, f"gerrit id: {id} does not exist")
            ret.append(GerritInfo(id, patch))
        return ret

    def apply_patches(self, patchs: List[str], checkout: bool) -> None:
        synapse = os.path.join(self.get_src_folder(), "synapse")
        self.org_hash = syn.git_hash(synapse)
        if self.base_revision:
            fetch_cmd = f"git -C {synapse} fetch"
            raise_if(
                0 != syn.run_external([fetch_cmd], None, LOG, True),
                "failed to fetch synapse",
            )
            checkout_cmd = f"git -C {synapse} checkout origin/{self.base_revision}"
            raise_if(
                0 != syn.run_external([checkout_cmd], None, LOG, True),
                f"failed to checkout base revision: {self.base_revision}",
            )
        gerrits = self.get_gerrits_infos(patchs)
        if gerrits:
            for gerrit in gerrits:
                ref = f"refs/changes/{gerrit.id[-2:]}/{gerrit.id}/{str(gerrit.patch)}"
                git_cmd = f"git -C {synapse} fetch origin {ref}"
                raise_if(
                    0 != syn.run_external([git_cmd], None, LOG, True),
                    f"failed to fetch gerrit id: {gerrit.id} to: {synapse}",
                )
                op = "checkout" if checkout else "cherry-pick"
                git_cmd = f"git -C {synapse} {op} FETCH_HEAD"
                raise_if(
                    0 != syn.run_external([git_cmd], None, LOG, True),
                    f"failed to {op} gerrit id: {gerrit.id} to: {synapse}",
                )
                LOG.info(
                    f"applyed gerrit id: {gerrit.id}, patch: {gerrit.patch} to: {synapse}"
                )


class QnpuStackGitRevs(QnpuStack):
    def __init__(
        self,
        work_folder_abs_path: str,
        qnpu_type: QnpuType,
        base_revision: str,
        remote: str,
        qnpu_version: str,
    ):
        super().__init__(
            work_folder_abs_path,
            qnpu_type,
            base_revision,
            qnpu_version,
        )
        self.remote: str = remote
        self.remote_name: str = "local"
        synapse = os.path.join(self.get_src_folder(), "synapse")
        git_cmd = f"git -C {synapse} remote remove {self.remote_name}"
        syn.run_external([git_cmd], None, LOG, True)
        git_cmd = f"git -C {synapse} remote add {self.remote_name} {self.remote}"
        raise_if(
            0 != syn.run_external([git_cmd], None, LOG, True),
            f"failed to add remote: {self.remote}",
        )
        git_cmd = f"git -C {synapse} fetch {self.remote_name}"
        raise_if(
            0 != syn.run_external([git_cmd], None, LOG, True),
            f"failed to fetch remote: {self.remote}",
        )

    def __del__(self):
        if self.remote:
            synapse = os.path.join(self.get_src_folder(), "synapse")
            git_cmd = f"git -C {synapse} remote remove {self.remote_name}"
            sts = syn.run_external([git_cmd], None, LOG, True)
            LOG.info(f"remote: {self.remote} remove status: {sts}")

    def apply_patches(self, patchs: List[str], checkout: bool) -> None:
        synapse = os.path.join(self.get_src_folder(), "synapse")
        self.org_hash = syn.git_hash(synapse)
        if self.base_revision:
            git_cmd = f"git -C {synapse} reset --hard"
            syn.run_external([git_cmd], None, LOG, True)
            git_cmd = f"git -C {synapse} fetch"
            syn.run_external([git_cmd], None, LOG, True)
            git_cmd = (
                f"git -C {synapse} checkout {self.remote_name}/{self.base_revision}"
            )
            raise_if(
                0 != syn.run_external([git_cmd], None, LOG, True),
                f"failed to checkout base revision: {self.base_revision}",
            )
        if patchs:
            for p in patchs:
                op = "checkout" if checkout else "cherry-pick"
                git_cmd = f"git -C {synapse} {op} {self.remote_name}/{p}"
                raise_if(
                    0 != syn.run_external([git_cmd], None, LOG, True),
                    f"failed to {op} git rev: {p} to: {synapse}",
                )
                LOG.info(f"applyed git rev: {p} to: {synapse}")


class Dispatcher:
    def __init__(self, pool_max_workers: int):
        self.pool_max_workers: int = pool_max_workers
        self.configs: List[ContainerConfig] = []
        self.consumers: List[ContainerConfig] = []

    @staticmethod
    def exe(config: ContainerConfig, delay: int = 0) -> int:
        time.sleep(delay)
        cmd = f'bash -c "hlctl create cr -f {config.yaml_path} --flavor {config.flavor} --retry"'
        config.status = syn.run_external([cmd], None, LOG, True)
        return config.status

    def put(self, config: ContainerConfig):
        self.configs.append(config)

    def put_consumer(self, config: ContainerConfig):
        self.consumers.append(config)

    def _dispatch(self, configs: List[ContainerConfig], pool_max_workers: int):
        tpe = ThreadPoolExecutor(pool_max_workers)
        delay = 0
        for c in configs:
            tpe.submit(Dispatcher.exe, c, delay)
            delay += 1
        tpe.shutdown(True)

    def dispatch(self):
        self._dispatch(self.configs, self.pool_max_workers)
        self._dispatch(self.consumers, 1)


class Launcher:
    def __init__(self):
        self.dispatchers: List[Dispatcher] = []

    def put(self, dispatcher: Dispatcher):
        self.dispatchers.append(dispatcher)

    def exe(self, dispatcher: Dispatcher):
        dispatcher.dispatch()

    def dispatch(self):
        tpe = ThreadPoolExecutor(len(self.dispatchers))
        for d in self.dispatchers:
            tpe.submit(self.exe, d)
        tpe.shutdown(True)


def get_device_list(names: List[str]):
    if names is None:
        return SUPPORTED_DEVICE
    ret = []
    for n in names:
        ret.append(DEVICES.get(NAME_TO_DEVICE_TYPE.get(n)))
    return ret


def create_min_xml(dst: str):
    manifest = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        "<manifest>\n",
        '<remote  name="gerrit" alias="origin" fetch="ssh://gerrit.habana-labs.com:29418" review="gerrit"/>\n',
        '<default revision="master" remote="gerrit" sync-j="4" />\n',
        '<project name="3rd-parties"/>\n',
        '<project name="habanalabs"/>\n',
        '<project name="automation"/>\n',
        '<project name="synapse"/>\n',
        "</manifest>",
    ]
    with open(dst, "w") as f:
        f.writelines(manifest)


def get_qnpu_tools(dst_folder: str):
    LOG.info("clone qnpu tools")
    qt_path = os.path.join(dst_folder, ".qnpu")
    if os.path.exists(qt_path):
        syn.git_pull(qt_path, None, LOG)
    else:
        syn.git_clone(QNPU_REPO, qt_path)


def qnpu_prerequisites(work_folder: str):
    get_qnpu_tools(work_folder)
    create_min_xml(os.path.join(work_folder, ".qnpu", "qnpu-manifests", "min.xml"))
    os.environ[
        "PATH"
    ] = f'{os.path.join(work_folder, ".qnpu")}:{os.environ.get("PATH")}'


def create_yaml(qnpu_version: str, container: dict):
    return {
        "apiVersion": "kubeflow.org/v1",
        "kind": "TFJob",
        "metadata": {
            "name": "tfjob",
            "annotations": {
                "habana.ai/hl_driver_version": f"{qnpu_version}",
                "repos": "mpm-test-data:master_next",
                "pod-reaper/max-duration": "4h",
            },
        },
        "spec": {
            "tfReplicaSpecs": {
                "Worker": {
                    "replicas": 1,
                    "template": {"spec": {"containers": [container]}},
                }
            }
        },
    }


def get_models_folder(run_config: RunConfig) -> Tuple[str, str]:
    models_folder = (
        "/root/models"
        if run_config.models_folder
        else "/root/repos/mpm-test-data/models"
    )
    if run_config.test_models:
        return "/root/repos/mpm-test-data/models", models_folder
    else:
        return models_folder, models_folder


def create_compile_yaml(
    dst: str,
    run_config: RunConfig,
    test_file_path: str,
    device: DeviceInfo,
    name: str,
):
    base_models_folder, candidate_models_folder = get_models_folder(run_config)
    models_folder = base_models_folder if name == "base" else candidate_models_folder
    container = {
        "name": "perf-test",
        "image": run_config.image_name,
        "command": [f"{test_file_path}"],
        "args": [
            f"{name}",
            f"{run_config.results_folder}",
            f"{device.name}",
            f"{models_folder}",
        ],
    }
    yaml_data = create_yaml(run_config.stack.get_version(), container)
    with open(dst, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None)


def create_run_yaml(
    dst: str,
    run_config: RunConfig,
    test_file_path: str,
    device: DeviceInfo,
    base_name: str,
    candidate_name: str,
):
    base_models_folder, candidate_models_folder = get_models_folder(run_config)
    container = {
        "name": "perf-test",
        "image": run_config.image_name,
        "command": [f"{test_file_path}"],
        "args": [
            f"{base_name}",
            f"{candidate_name}",
            f"{run_config.results_folder}",
            f"{device.name}",
            f"{base_models_folder}",
            f"{candidate_models_folder}",
        ],
    }
    yaml_data = create_yaml(run_config.stack.get_version(), container)
    with open(dst, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None)


def create_report_yaml(
    dst: str,
    image: str,
    qnpu_version: str,
    output_folder: str,
    test_file_path: str,
    device: DeviceInfo,
    base_name: str,
):
    container = {
        "name": "perf-test",
        "image": image,
        "command": [f"{test_file_path}"],
        "args": [
            f"{base_name}",
            f"{output_folder}",
            f"{device.name}",
        ],
    }
    yaml_data = create_yaml(qnpu_version, container)
    with open(dst, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None)


def create_compile_test_exe(test_file_name: str, run_config: RunConfig):
    pull_cmd = f"git -C /root/repos/mpm-test-data lfs pull && git -C /root/repos/mpm-test-data checkout {run_config.models_revision}"
    jobs = f"--jobs {' '.join(run_config.jobs)}" if run_config.jobs else ""
    cmd = [
        "#!/bin/bash\n",
        f"set -x\n",
        f"NAME=$1\n",
        f"FOLDER=$2\n",
        f"DEVICE=$3\n",
        f"MODELS_FOLDER=$4\n",
        f"{pull_cmd}\n",
        f'bash -ic "LD_PRELOAD=/root/builds/synapse_release_build/lib/libSynapse.$NAME.so run_models_tests --max_mem_limit {str(int(32e9))} --log_level d -c $DEVICE --threads {run_config.threads} --models_folder $MODELS_FOLDER --work_folder $FOLDER/$DEVICE --actions compile --report_path $FOLDER/results_$DEVICE -n comp-$NAME {jobs}" || true\n',
    ]
    dst = os.path.join(run_config.docker_folder, test_file_name)
    with open(dst, "w") as f:
        f.writelines(cmd)
    syn.run_external(["chmod", "+x", dst], None, LOG, True)


def create_run_test_exe(test_file_name: str, run_config: RunConfig):
    pull_cmd = f"git -C /root/repos/mpm-test-data lfs pull && git -C /root/repos/mpm-test-data checkout {run_config.models_revision}"
    jobs = f"--jobs {' '.join(run_config.jobs)}" if run_config.jobs else ""
    cmd = [
        "#!/bin/bash\n",
        f"set -x\n",
        f"BASE=$1\n",
        f"CANDIDATE=$2\n",
        f"FOLDER=$3\n",
        f"DEVICE=$4\n",
        f"BASE_MODELS_FOLDER=$5\n",
        f"CANDIDATE_MODELS_FOLDER=$6\n",
        f"{pull_cmd}\n",
    ]
    if not run_config.run_once:
        cmd += [
            f'bash -ic "LD_PRELOAD=/root/builds/synapse_release_build/lib/libSynapse.$BASE.so run_models_tests --max_mem_limit {str(int(32e9))} --log_level d -c $DEVICE --models_folder $BASE_MODELS_FOLDER --work_folder $FOLDER/$DEVICE --actions run --report_path $FOLDER/results_$DEVICE -p comp-$BASE -n $BASE {jobs}" || true\n',
        ]
    cmd += [
        f'bash -ic "LD_PRELOAD=/root/builds/synapse_release_build/lib/libSynapse.$CANDIDATE.so run_models_tests --max_mem_limit {str(int(32e9))} --log_level d -c $DEVICE --models_folder $CANDIDATE_MODELS_FOLDER --work_folder $FOLDER/$DEVICE --actions run --report_path $FOLDER/results_$DEVICE -p comp-$CANDIDATE -n $CANDIDATE {jobs}" || true\n',
    ]
    dst = os.path.join(run_config.docker_folder, test_file_name)
    with open(dst, "w") as f:
        f.writelines(cmd)
    syn.run_external(["chmod", "+x", dst], None, LOG, True)


def create_report_test_exe(dst: str, run_config: RunConfig):
    models_folder = (
        "/root/models"
        if run_config.models_folder
        else "/root/repos/mpm-test-data/models"
    )
    cmd = [
        "#!/bin/bash\n",
        f"set -x\n",
        f"BASE=$1\n",
        f"FOLDER=$2\n",
        f"DEVICE=$3\n",
        f'bash -ic "run_models_tests --log_level d --models_folder {models_folder} --work_folder $FOLDER/$DEVICE --actions report --ref_name $BASE --threshold {REGRESSION_THRESHOLD} --report_path $FOLDER/results_$DEVICE"\n',
    ]
    with open(dst, "w") as f:
        f.writelines(cmd)
    syn.run_external(["chmod", "+x", dst], None, LOG, True)


def create_docker_file(dst: str, run_config: RunConfig):
    copy_models_cmd = "COPY models /root/models" if run_config.models_folder else ""
    docker = [
        "FROM artifactory-kfs.habana-labs.com/developers-docker-dev-local/gc/base:ubuntu20.04\n",
        "WORKDIR /root/\n",
        "COPY .bash* /root/\n",
        "COPY *.sh /root/\n",
        "COPY src/automation /root/trees/npu-stack/automation\n",
        "COPY src/synapse /root/trees/npu-stack/synapse/\n",
        "COPY bin /root/builds/\n",
        f"{copy_models_cmd}\n",
        'ENV PATH="$PATH:/root/builds/latest"\n',
        "RUN apt-get update && apt-get install -y ssh git-lfs mailutils",
    ]
    with open(dst, "w") as f:
        f.writelines(docker)


def create_bashrc_file(dst: str):
    bashrc = [
        "unset SET_ABSOLUTE_HABANA_ENV\n",
        'source "$HOME"/trees/npu-stack/automation/habana_scripts/habana_env',
    ]
    with open(dst, "w") as f:
        f.writelines(bashrc)


def get_models(dst_folder: str, source_folder: str):
    LOG.info("update models folder")
    models_folder_path = (
        os.path.abspath(f"{LOCAL_MODELS_FOLDER}/models")
        if source_folder is None
        else source_folder
    )
    if source_folder is None:
        if os.path.exists(f"{LOCAL_MODELS_FOLDER}/.git"):
            syn.git_pull(LOCAL_MODELS_FOLDER)
        else:
            os.makedirs(LOCAL_MODELS_FOLDER, exist_ok=True)
            syn.git_clone(MODELS_REPO, LOCAL_MODELS_FOLDER)
            syn.git_checkout(LOCAL_MODELS_FOLDER, MODELS_REVISION)
    shutil.copytree(
        models_folder_path,
        os.path.join(dst_folder, "models"),
        symlinks=False,
    )


def create_common_docker_content(docker_folder: str, models_folder: str):
    os.makedirs(docker_folder, exist_ok=True)
    create_bashrc_file(os.path.join(docker_folder, ".bashrc"))
    get_models(docker_folder, models_folder)


def copy(src: str, dst: str, ext: str):
    files = glob.iglob(os.path.join(src, f"*.{ext}"))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dst)


def create_compile_test_files(
    run_config: RunConfig,
) -> Dict[DeviceType, List[ContainerConfig]]:
    ret = {}
    test_file = "compile.sh"
    create_compile_test_exe(test_file, run_config)
    for device in run_config.devices:
        tests = []
        steps = ["candidate"] if run_config.run_once else ["base", "candidate"]
        for test in steps:
            yaml_path = os.path.join(
                run_config.docker_folder, f"compile-{test}-{device.name}.yaml"
            )
            create_compile_yaml(
                yaml_path,
                run_config,
                os.path.join("/root", test_file),
                device,
                test,
            )
            tests.append(ContainerConfig(yaml_path, device.compile_container))
        ret[device] = tests
    return ret


def create_run_test_files(run_config: RunConfig) -> Dict[DeviceType, ContainerConfig]:
    ret = {}
    test_file = "run.sh"
    create_run_test_exe(test_file, run_config)
    for device in run_config.devices:
        yaml_path = os.path.join(run_config.docker_folder, f"run-{device.name}.yaml")
        create_run_yaml(
            yaml_path,
            run_config,
            os.path.join("/root", test_file),
            device,
            "base",
            "candidate",
        )
        ret[device] = ContainerConfig(yaml_path, device.run_container)
    return ret


def create_report_test_files(
    run_config: RunConfig,
) -> Dict[DeviceType, ContainerConfig]:
    ret = {}
    test_file = "report.sh"
    create_report_test_exe(
        os.path.join(run_config.docker_folder, test_file), run_config
    )
    for device in run_config.devices:
        yaml_path = os.path.join(run_config.docker_folder, f"report-{device.name}.yaml")
        create_report_yaml(
            yaml_path,
            run_config.image_name,
            run_config.stack.get_version(),
            run_config.results_folder,
            os.path.join("/root", test_file),
            device,
            "base",
        )
        ret[device] = ContainerConfig(yaml_path, device.compile_container)
    return ret


def build_image(run_config: RunConfig):
    LOG.info("copy binaries")
    os.makedirs(os.path.join(run_config.docker_folder, "bin", "latest"), exist_ok=True)
    copy(
        os.path.join(run_config.stack.get_bin_folder(), "latest"),
        os.path.join(run_config.docker_folder, "bin", "latest"),
        "so",
    )
    shutil.copy2(
        os.path.join(run_config.stack.get_bin_folder(), "latest", "llvm-objcopy"),
        os.path.join(run_config.docker_folder, "bin", "latest", "llvm-objcopy"),
        follow_symlinks=True,
    )
    shutil.copy2(
        os.path.join(run_config.stack.get_bin_folder(), "latest", "llvm-objdump"),
        os.path.join(run_config.docker_folder, "bin", "latest", "llvm-objdump"),
        follow_symlinks=True,
    )
    shutil.copy2(
        os.path.join(run_config.stack.get_bin_folder(), "latest", "tpc-clang"),
        os.path.join(run_config.docker_folder, "bin", "latest", "tpc-clang"),
        follow_symlinks=True,
    )
    shutil.copytree(
        os.path.join(run_config.stack.get_bin_folder(), "synapse_release_build", "lib"),
        os.path.join(run_config.docker_folder, "bin", "synapse_release_build", "lib"),
        symlinks=False,
        ignore=shutil.ignore_patterns("latest", "*debug*", "*.a", "*Validation*"),
    )
    shutil.copytree(
        os.path.join(run_config.stack.get_bin_folder(), "synapse_release_build", "bin"),
        os.path.join(run_config.docker_folder, "bin", "synapse_release_build", "bin"),
        symlinks=False,
        ignore=shutil.ignore_patterns("latest"),
    )
    shutil.copytree(
        os.path.join(run_config.stack.get_bin_folder(), "engines_fw_release_build"),
        os.path.join(run_config.docker_folder, "bin", "engines_fw_release_build"),
        symlinks=False,
        ignore=shutil.ignore_patterns("CMake", "fw_tests"),
    )
    tpc_kenels_dst = os.path.join(
        run_config.docker_folder, "bin", "tpc_kernels_release_build", "src"
    )
    os.makedirs(tpc_kenels_dst, exist_ok=True)
    shutil.copy(
        os.path.join(
            run_config.stack.get_bin_folder(),
            "tpc_kernels_release_build",
            "src",
            "libtpc_kernels.so",
        ),
        os.path.join(tpc_kenels_dst, "libtpc_kernels.so"),
    )
    shutil.copytree(
        os.path.join(run_config.stack.get_src_folder(), "automation"),
        os.path.join(run_config.docker_folder, "src", "automation"),
        symlinks=False,
        ignore=shutil.ignore_patterns(".git"),
    )
    run_config.stack.apply_patches(run_config.base_patches + run_config.test_patchs, run_config.checkout)
    shutil.copytree(
        os.path.join(run_config.stack.get_src_folder(), "synapse"),
        os.path.join(run_config.docker_folder, "src", "synapse"),
        symlinks=False,
    )
    run_config.stack.revert()
    docker_file_name = "gc-perf.dockerfile"
    create_docker_file(
        os.path.join(run_config.docker_folder, docker_file_name), run_config
    )
    os.chdir(run_config.docker_folder)
    syn.run_external(
        ["docker", "build", "-f", docker_file_name, "-t", run_config.image_name, "."],
        None,
        LOG,
        True,
    )
    syn.run_external(
        ["docker", "image", "push", run_config.image_name], None, LOG, True
    )
    syn.run_external(["docker", "image", "rm", run_config.image_name], None, LOG, True)


def create_container(
    run_config: RunConfig,
) -> Tuple[
    Dict[DeviceType, List[ContainerConfig]],
    Dict[DeviceType, ContainerConfig],
    Dict[DeviceType, ContainerConfig],
]:
    if os.path.exists(run_config.docker_folder):
        shutil.rmtree(run_config.docker_folder)

    create_common_docker_content(run_config.docker_folder, run_config.models_folder)

    if run_config.base_patches is None:
        run_config.base_patches = []
    if run_config.test_patchs is None:
        run_config.test_patchs = []

    if not run_config.run_once:
        run_config.stack.apply_patches_and_build(
            "base", run_config.base_patches, run_config.checkout
        )
    run_config.stack.apply_patches_and_build(
        "candidate",
        run_config.base_patches + run_config.test_patchs,
        run_config.checkout,
    )

    compile_yamls = create_compile_test_files(run_config)
    run_yamls = create_run_test_files(run_config)
    report_yamls = create_report_test_files(run_config)
    build_image(run_config)
    return compile_yamls, run_yamls, report_yamls


def send_status_mail(subject: str, message: str, attachments: List[str] = None):
    attachments_arg = f"-A {' '.join(attachments)}" if attachments else ""
    email_cmd = [
        f"echo '{message}' | mail -s '{subject}' {getpass.getuser()}@habana.ai {attachments_arg}",
    ]
    syn.run_external(email_cmd, None, LOG, False)


def run(args) -> int:
    os.makedirs(args.work_folder, exist_ok=True)
    qnpu_prerequisites(args.work_folder)

    run_config = RunConfig(args)
    compile_yamls, run_yamls, report_yamls = create_container(run_config)

    launcher = Launcher()
    for k, v in compile_yamls.items():
        dispatcher = Dispatcher(1 if run_config.run_once else 2)
        for t in v:
            dispatcher.put(t)
        if k in run_yamls:
            dispatcher.put_consumer(run_yamls.get(k))
        launcher.put(dispatcher)
    launcher.dispatch()

    sts: int = 0
    for d in launcher.dispatchers:
        for c in d.configs:
            sts |= c.status
            if c.status != 0:
                LOG.error(
                    f"container run failed, yaml: {c.yaml_path}, flavor: {c.flavor}, status: {c.status}"
                )

    for c in report_yamls.values():
        report_status = Dispatcher.exe(c)
        sts |= report_status
        if report_status != 0:
            LOG.error(
                f"container run failed, yaml: {c.yaml_path}, flavor: {c.flavor}, status: {c.status}"
            )

    LOG.info(f"results written to {args.results_folder}")

    return sts


def verify_sufficient_free_disk_space(required_disk_space: int):
    total, used, free = shutil.disk_usage("/")
    total = total // (2**30)
    used = used // (2**30)
    free = free // (2**30)
    if free < required_disk_space:
        raise RuntimeError(
            f"insufficient free disk space: total: {total}[GB], used: {used}[GB], free: {free}[GB]{os.linesep}minimum required free disk space: {required_disk_space}[GB]"
        )


def artifactory_login():
    registry = "artifactory-kfs.habana-labs.com"
    login_required: bool = False

    docker_config = os.path.abspath(
        os.path.join(os.path.expanduser("~"), ".docker", "config.json")
    )

    if not os.path.exists(docker_config):
        login_required = True
    else:
        with open(docker_config) as f:
            j = json.load(f)
            auths = j.get("auths")
            login_required = True if auths is None or registry not in auths else False

    login_cmd = ["docker", "login", registry]
    if login_required:
        login_cmd += ["--username", getpass.getuser()]
        LOG.info(
            f"{' '.join(login_cmd)}{os.linesep}Login to artifactory, this should be done once{os.linesep}Enter password:"
        )

    raise_if(
        syn.run_external(login_cmd, None, LOG, False) != 0,
        f"failed to login to {registry}",
    )


def main(args):
    os.makedirs(args.results_folder, exist_ok=True)
    log_path = os.path.join(
        args.results_folder, f'perf-test-{time.strftime("%Y%m%d-%H%M%S")}.log'
    )
    mail_subject = ""
    mail_body = ""
    try:
        config_logger(log_path)
        verify_sufficient_free_disk_space(40)
        artifactory_login()
        sts = run(args)
        if sts == 0:
            mail_subject = "Models tests results are ready"
            mail_body = (
                f"Models test results are waiting for you at: {args.results_folder}"
            )
        else:
            mail_subject = "Models tests run failed"
            mail_body = f"Test run failed, results are waiting for you at: {args.results_folder}"
    except Exception as e:
        mail_body = f"Test run failed, command: {' '.join(sys.argv)},  error: {e}"
        traceback.print_exc()
        print(mail_body)
    finally:
        if not mail_subject:
            mail_subject = "Models tests failed to run"
        if not mail_body:
            mail_body = f"Test run failed, command: {' '.join(sys.argv)}"
        send_status_mail(mail_subject, mail_body)
        shutil.rmtree(
            os.path.abspath(os.path.join(args.work_folder, "temp")), ignore_errors=True
        )
        shutil.rmtree(
            os.path.abspath(os.path.join(args.work_folder, "docker")),
            ignore_errors=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-w",
        "--work-folder",
        help=argparse.SUPPRESS,
        default=os.path.abspath(os.path.join(os.path.expanduser("~"), "perfcheck")),
    )
    parser.add_argument(
        "--results-folder",
        help="Results folder path, must be accessible from container run",
        default=f'/software/users/{getpass.getuser()}/perf-test/{time.strftime("%Y%m%d-%H%M%S")}',
    )
    compare_select_group = parser.add_mutually_exclusive_group()
    compare_select_group.add_argument(
        "--run-once",
        action="store_true",
        help="Run only the --test-patches without perf compare",
    )
    compare_select_group.add_argument(
        "--test-models",
        action="store_true",
        help="Run two differnet models set versions, if enabled --models-folder should point to the tested models folder",
    )
    compare_select_group.add_argument(
        "-b",
        "--base-patches",
        help="Base patchs, list of patches to use for base and tested synapse libaray, the patches are rebased over --base-rev. "
        "If not set the test will run once without perf comparison",
        nargs="+",
    )
    parser.add_argument(
        "-t",
        "--test-patches",
        help="Tested patchs, list of patches to use for tested synapse libaray, the patches are rebased over --base-rev + --base-patches",
        nargs="+",
    )
    parser.add_argument(
        "--gerrit",
        action="store_true",
        help="Both --test-patches and --base-patchs points to gerrit IDs",
    )
    parser.add_argument(
        "--remote",
        help="Remote repo for git rev patchs",
        default=os.environ.get("SYNAPSE_ROOT"),
    )
    parser.add_argument(
        "--qnpu-version",
        help="Use specific QNPU version instead of latest",
    )
    models_group = parser.add_mutually_exclusive_group()
    models_group.add_argument(
        "--models-folder",
        help="Use local models folder instead cloning one",
    )
    models_group.add_argument(
        "--models-revision",
        help="Models git revision",
        default="master_next",
    )
    parser.add_argument(
        "-c",
        "--chip-types",
        nargs="+",
        help="Select tested devices",
        choices=[d.name for d in SUPPORTED_DEVICE],
        default=[d.name for d in SUPPORTED_DEVICE],
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        help="Select tested devices",
        choices=[
            "post-submit",
            "dynamic",
            "promotion",
            "compilation",
            "accuracy",
            "debug",
        ],
    )
    parser.add_argument(
        "--base-rev",
        help="Apply patchs upon specific git revision (hash/branch/tag), revision must be available at --remote repo",
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        help="Checkout instead of cherry-pick",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Limit the number of compilation threads, 0 for no limit",
        default=0,
    )
    args = parser.parse_args()
    if args.checkout and args.test_patches and len(args.test_patches) > 1:
        parser.error(f"checkout is enabled, only single --test-patches is allowed")
    if args.checkout and args.base_patches and len(args.base_patches) > 1:
        parser.error(f"checkout is enabled, only single --base-patchs is allowed")
    if args.test_models:
        args.base_patches = args.test_patches
    args.work_folder = os.path.join(args.work_folder)

    return args


if __name__ == "__main__":
    args = parse_args()
    exit(main(args))
