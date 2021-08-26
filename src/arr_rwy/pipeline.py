# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction of the master pipeline.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from arr_rwy.pipelines import data_query_and_save as dqs
from arr_rwy.pipelines import data_engineering as de
from arr_rwy.pipelines import data_science as ds


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project"s pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    dqs_pipeline = dqs.create_pipelines()
    de_pipelines = de.create_pipelines()
    ds_pipeline = ds.create_pipelines()

    pipelines = {}
    pipelines["dqs"] = dqs_pipeline
    pipelines["de_tv"] = de_pipelines["de_tv"]
    pipelines["de_ntv"] = de_pipelines["de_ntv"]
    pipelines["de"] = de_pipelines["de_overall"]
    pipelines["ds"] = ds_pipeline
    pipelines["all"] = pipelines["dqs"] + pipelines["de"] + pipelines["ds"]

    pipelines["__default__"] = pipelines["all"]

    return pipelines
