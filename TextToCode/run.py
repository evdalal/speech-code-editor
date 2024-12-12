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

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)
