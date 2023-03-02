

import sys
import os

sys.path.append('../')

from old_benchmark.SLCT_benchmark import benchmark_settings
from logparser.SLCT import SLCT
from evaluation.utils.common import datasets, common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results, prepare_results2

input_dir = '../logs/'  # The input directory of log file
output_dir = unique_output_dir('SLCT')  # The output directory of parsing result

if __name__ == "__main__":

    args = common_args()

    # prepare result_file
    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction
    )
    #output_dir=input_dir+"/output"
    res2 = prepare_results2(os.path.join(input_dir+"/output", result_file+'_slct__accuracy.csv'))
    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file']
        indir = os.path.join(input_dir, os.path.dirname(log_file))

        # run evaluator for a dataset
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=SLCT.LogParser,
            param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex'],
                'support': setting['support']
            },
            otc=args.oracle_template_correction,
            result_file=res2
        )  # it internally saves the results into a summary file
