# Copyright 2020 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from .models import (get_class_fit,
                     create_class,
                     create_function,
                     gen_sklearn_model,
                     eval_class_model)

from .plots import (gcf_clear,
                    feature_importances,
                    learning_curves,
                    confusion_matrix,
                    precision_recall_multi,
                    roc_multi,
                    roc_bin,
                    precision_recall_bin)

from .data import (get_sample,
                   get_splits,
                   save_heldout)

__version__ = '0.3.0'