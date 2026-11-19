"""CPU-only faNN format definitions; no model or VTK imports."""
import pickle

FIELDS = ("VelocityX", "VelocityY", "VelocityZ", "Pressure", "Temperature", "Density")

class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
                   ('numpy', 'dtype'), ('numpy.core.numeric', '_frombuffer')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')
        return super().find_class(module, name)

