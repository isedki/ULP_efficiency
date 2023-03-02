
import sys
import os

sys.path.append('../')

from old_benchmark import AEL_benchmark
from logparser.AEL import LogParser
from evaluation.utils.common import datasets, common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results, prepare_results2

input_dir = '../logs/efficiency'  # The input directory of log file
output_dir = unique_output_dir('AEL')  # The output directory of parsing result

if __name__ == "__main__":

    args = common_args()

    # prepare result_file
    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction
    )
    
    #output_dir=input_dir+"/output"
    res2 = prepare_results2(os.path.join(input_dir+"/output", 'ael_efficiency.csv'))
    tab = ['5k','10k','20k','40k','60k','80k','100k']

    for dataset in datasets:
        setting = AEL_benchmark.benchmark_settings[dataset]
        for name in tab:
            log_file = dataset+name+".txt"
            indir = os.path.join(input_dir, os.path.dirname(log_file))

        # run evaluator for a dataset
            evaluator(
                dataset=dataset,
                size=name,
                input_dir=input_dir,
                output_dir=output_dir,
                log_file=log_file,
                LogParser=LogParser,
                param_dict={
                'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir, 'rex': setting['regex'],
                'minEventCount': setting['minEventCount'], 'merge_percent': setting['merge_percent']
            },
            otc=args.oracle_template_correction,
            result_file=res2
        )  # it internally saves the results into a summary file
