from mlvt.server.views.base import BaseView


class ConfigsView(BaseView):
    def get(self, name):
        config = self._get_config(name)
        if config is None:
            return f'Configuration with name ({name}) not found', 404

        return config, 200

    def search(self):
        configs = []
        for config_name in self.cm.configurations:
            configs.append(self._get_config(config_name))
        return configs, 200

    def _get_config(self, name):
        if name not in self.cm.configurations:
            return

        config = self.cm.get_config()
        config['name'] = name
        return config
