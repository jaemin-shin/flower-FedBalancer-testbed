#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

pyenv virtualenv 3.7.12 fedbalancer-3.7.12
echo fedbalancer-3.7.12 > .python-version
