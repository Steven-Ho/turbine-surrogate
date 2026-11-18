from typing import Dict

from .field_processor import FieldProcessor


class ImageProcessor(FieldProcessor):
    """Reserved processor that is not enabled in the current mainline."""

    # Defines `__init__` for   init   logic.
    def __init__(self, config: Dict):
        super().__init__(config)
        raise NotImplementedError(
            "ImageProcessor is not enabled in the current mainline. "
            # Completes the multi-line `NotImplementedError` arguments or elements.
            "Use ImageVoxelProcessor for field tasks with UNet-like models."
        )
