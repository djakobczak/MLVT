from copy import deepcopy

from mlvt.config.config_manager import ConfigManager
from mlvt.server.config import CURRENT_CONFIG_FILE
from mlvt.server.views.base import BaseView


class ConfigsView(BaseView):
    def get(self, name):
        self.init_cm()
        config = self._get_config(name)
        if config is None:
            return f'Configuration with name ({name}) not found', 404

        return config, 200

    def search(self):
        self.init_cm()
        configs = []
        for config_name in self.cm.configurations:
            configs.append(self._get_config(config_name))
        return configs, 200

    def _get_config(self, name):
        if name not in self.cm.configurations:
            return

        config = deepcopy(self.cm.get_config(name))
        config['name'] = name
        return config

    def put(self, name):
        if name not in ConfigManager.configurations:
            return 'Config file not found', 404

        with open(CURRENT_CONFIG_FILE, 'w') as f:
            f.write(name)

        return 'New config set', 200
