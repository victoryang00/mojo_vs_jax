#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

set -e

CURRENT_DIR=$(dirname "$0")
# Make sure we're running from inside the directory containing this file.
cd "$CURRENT_DIR"
MAX_PKG_DIR="${MAX_PKG_DIR:-$CONDA_PREFIX}"
export MAX_PKG_DIR
# Build the example
cmake -B build -S "$CURRENT_DIR"
cmake --build build

# Run example
./build/mlp
