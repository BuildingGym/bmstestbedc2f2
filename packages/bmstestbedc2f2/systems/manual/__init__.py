
# TODO
from controllables.core import MutableVariable
from controllables.core.systems import SimpleProcess, ProtoProcess
from bmstestbedc2f2.systems.common import ProtoBMSystem, _TODO_ProtoBMSystem


class ManualBMSystem(SimpleProcess, ProtoBMSystem):
    def __init__(self):
        super().__init__(slots=self.__variable_slots__)


# TODO !!!!!!!
class _TODO_ManualBMSystem(SimpleProcess, _TODO_ProtoBMSystem):
    def __init__(self):
        super().__init__()

        for slot in self.__variable_slots__:
            self.variables._variables[slot] = MutableVariable()

        for slot, var in self._basevars.items():
            self.variables._variables[slot] = var

