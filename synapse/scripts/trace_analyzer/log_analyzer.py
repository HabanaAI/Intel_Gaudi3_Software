import re
from config import LOG_INFO_PARTIAL_INFO_KEY, LOG_INFO_CACHE_WARMUP_NODE_NAMES
class LogAnalyzer:
    def __init__(self, log_path):
        self.log_path = log_path
        self.nodes_info = {}
        self._parse()

    def _parse(self):
        with open(self.log_path, 'r') as log_file:
            node_info = []
            node_name = ''
            for line in log_file:
                match = re.match(r'.*=> (.*)', line)
                if match:
                    # if node_info:
                    #     print(f"error(no node name): {self.log_path},{node_info} without node name ")
                    if not node_name:
                        node_info.append(match.group(1))
                    else:
                        node_info = [match.group(1)]
                        node_name = ''
                else:
                    match = re.match(r'.*Node (.*) (.*)$', line)
                    if match:
                        if not node_info:
                            print(f"error(no info above): {self.log_path},{match.group(1)} without node info ")
                        else:
                            node_name = match.group(1)
                            if node_name in self.nodes_info:
                                # print(f"error(info exists): {self.log_path},{node_name} exists. {self.nodes_info[node_name]},node_info")
                                self.nodes_info[node_name][LOG_INFO_PARTIAL_INFO_KEY].extend(node_info)
                            else:
                                self.nodes_info[node_name] = {LOG_INFO_PARTIAL_INFO_KEY: node_info}
                    else:
                        match = re.match(r'.*Add (cache_warmup_.*) \(.*\)$', line)
                        if match:
                            if node_name and node_name in self.nodes_info:
                                if 'warmup' in self.nodes_info[node_name]:
                                    self.nodes_info[node_name][LOG_INFO_CACHE_WARMUP_NODE_NAMES].append(match.group(1))
                                else:
                                    self.nodes_info[node_name][LOG_INFO_CACHE_WARMUP_NODE_NAMES] = [match.group(1)]
                            else:
                                print(f"error: {line}")

    def get_nodes_log_info(self):
        return self.nodes_info