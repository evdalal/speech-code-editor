"""
# Project Name: Speech to Code
# Author: STC Team
# Date: 12/12/2024
# Last Modified: 12/12/2024
# Version: 1.0

# Copyright (c) 2024 Brown University
# All rights reserved.

# This file is part of the STC project.
# Usage of this file is restricted to the terms specified in the
# accompanying LICENSE file.

"""

class OllamaConnectionError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class OllamaModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class OllamaResourceNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

