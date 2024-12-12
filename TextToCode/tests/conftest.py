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

import pytest
from app import create_app

@pytest.fixture
def mock_requests_post(mocker):
    # This fixture sets up the mock for requests.post
    def _mock_requests_post(status_code=200, json_data=None, raise_exception=None):
        mock_response = mocker.Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data
        mock_response.text = json_data

        if raise_exception:
            mocker.patch('app.views.requests.post', side_effect=raise_exception)
        else:
            mocker.patch('app.views.requests.post', return_value=mock_response)
        
        return mock_response

    return _mock_requests_post

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client