import json
from enum import Enum

from mpm_types import ModelStats, MpmObj, Stats
from typing import Dict, List, Optional, Set


class Filter(MpmObj):
    def __init__(self, device: str = None, job: str = None, check_disabled: bool = False):
        self.device: str = device
        self.job: str = job
        self.check_disabled: bool = check_disabled


class ModelsDeviceInfo(MpmObj):
    def __init__(self):
        self.jobs: List[str] = []
        self.stats: ModelStats = ModelStats()
        self.disabled: bool = False

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        ret.from_dict(data)
        if data:
            stats = data.get("stats", {})
            ret.stats = ModelStats.deserialize(stats)
            ret.jobs = data.get("jobs", [])
            ret.disabled = data.get("disabled", False)
        return ret


class ModelMetadata(MpmObj):
    def __init__(self, description: str = ""):
        self.description: str = description
        self.devices: Dict[str, ModelsDeviceInfo] = {}
        self.disabled: bool = False

    def add(self, filter: Filter = None, description: str = None):
        if filter is None or filter.device is None:
            return
        if description:
            self.description = description
        device: ModelsDeviceInfo = self.devices.setdefault(
            filter.device, ModelsDeviceInfo()
        )
        if filter.job:
            if filter.job not in device.jobs:
                device.jobs.append(filter.job)

    def remove(self, filter: Filter):
        if filter is None or filter.device is None or filter.device not in self.devices:
            return
        if filter.job:
            filters = self.devices.get(filter.device)
            if filters and filter.job in filters.jobs:
                filters.jobs.remove(filter.job)
        else:
            self.devices.pop(filter.device)

    def should_skip(self, filter: Filter) -> bool:
        if filter is None:
            return False
        if self.disabled and filter.check_disabled:
            return True
        if filter.device is None:
            return False
        if filter.device in self.devices:
            filters = self.devices.get(filter.device)
            if filters.disabled and filter.check_disabled:
                return True
            if not filters or filter.job is None:
                return False
            if filters and filter.job in filters.jobs:
                return False
        return True

    def get_jobs(self) -> Set[str]:
        ret = set()
        for v in self.devices.values():
            for j in v.jobs:
                ret.add(j)
        return ret

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        ret.from_dict(data)
        if data:
            devices = data.get("devices", {})
            for k, v in devices.items():
                ret.devices[k] = ModelsDeviceInfo.deserialize(v)
            ret.disabled = data.get("disabled", False)
        return ret


class ModelsList(MpmObj):
    def __init__(self):
        self.version: int = 1
        self.models: Dict[str, ModelMetadata] = {}

    def add(self, model_name: str, model_info: ModelMetadata) -> None:
        self.models[model_name] = model_info

    def remove(self, model_name: str) -> None:
        self.models.pop(model_name)

    def get_info(self, model: str) -> Optional[ModelMetadata]:
        return self.models.get(model)

    def get_names(self, filter: Filter) -> List[str]:
        ret: List[str] = []
        for k, v in self.models.items():
            if not v.should_skip(filter):
                ret.append(k)
        return ret

    def get_jobs(self) -> Set[str]:
        ret = set()
        for v in self.models.values():
            ret.update(v.get_jobs())
        return ret

    @classmethod
    def deserialize(cls, data):
        ret = cls()
        if data and data.get("models"):
            for k, v in data.get("models").items():
                ret.models[k] = ModelMetadata.deserialize(v)
        return ret
