"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os

sys.path.append('../')


from old_benchmark.ULP_benchmark import benchmark_settings
from logparser.ULP import ULP
from evaluation.utils.common import datasets, common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results

input_dir = '/logs2/'  # The input directory of log file
output_dir = './output'  # The output directory of parsing result

if __name__ == "__main__":

    args = common_args()

    # prepare result_file
    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction
    )

    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file']
        #indir = os.path.join(input_dir, os.path.dirname(log_file))
        indir= input_dir

        # run evaluator for a dataset
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=ULP.LogParser,
            param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex']
            },
            otc=args.oracle_template_correction,
            result_file="summary_ulp"
        )  # it internally saves the results into a summary file