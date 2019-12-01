import copy
import uuid
from datetime import datetime, timezone
import jsons

DATETIME_FORMAT = "%m-%d-%Y, %H:%M:%S"
SCHEMA_VERSION = '1.0.0'


class BaseReport:
    @staticmethod
    def from_dict(dict_: dict):
        return jsons.load(dict_, BaseReport)

    def __init__(self, correlation_id=None):
        self.execution_id = str(uuid.uuid4())
        self.correlation_id = correlation_id if correlation_id is not None else self.execution_id
        self.start_time = datetime.now(timezone.utc)
        self.data = {}
        self.finish_time = None
        self.duration = None

    def to_dict(self):
        result = copy.deepcopy(self.__dict__)
        result['start_time'] = str(self.start_time)
        result['finish_time'] = str(self.finish_time)
        result['duration'] = str(self.duration)
        return jsons.dump(result)

    def add(self, name: str, value):
        setattr(self, name, value)

    def add_data(self, name: str, value):
        self.data[name] = value

    def get_data(self, name: str):
        return self.data.get(name)

    def get(self, name: str):
        return getattr(self, name)

    def set_parameters(self, parameters):
        for k, v in parameters.items():
            setattr(self, k, v)

    def set_finish_time(self):
        self.finish_time = datetime.now(timezone.utc)
        self.duration = str(self.finish_time - self.start_time)
