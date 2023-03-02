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

import os
import time
import csv

from multiprocessing import Process
from evaluation.utils.common import correct_templates_and_update_files
from logparser.utils.evaluator import evaluate
from evaluation.utils.template_level_analysis import evaluate_template_level,evaluate_template_level_hdfs
from evaluation.utils.PA_calculator import calculate_parsing_accuracy

TIMEOUT = 3600  # log template identification timeout (sec)


def prepare_results(output_dir, otc):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'summary_'.format(str(otc))
    with open(os.path.join(output_dir, result_file), 'w') as csv_file:
        fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['Dataset', 'GA_time', 'PA_time', 'TA_time', 'parse_time', 'identified_templates',
                     'ground_templates', 'GA', 'PA', 'FTA', 'PTA', 'RTA', 'OG', 'UG', 'MX'])

    return result_file


def evaluator(
        dataset,
        input_dir,
        output_dir,
        log_file,
        LogParser,
        param_dict,
        otc,
        result_file
):
    """
    Unit function to run the evaluation for a specific configuration.

    """

    print('\n=== Evaluation on %s ===' % dataset)
    #indir = os.path.join(input_dir, os.path.dirname(log_file))
    indir= input_dir
    log_file_basename = os.path.basename(log_file)


    # identify templates using Drain
    start_time = time.time()
    parser = LogParser(**param_dict)
    p = Process(target=parser.parse, args=(log_file_basename,))
    #p = Process(target=parser.Process_Remove_Duplicates, args=(log_file, "", 2000))
    p.start()
    p.join(timeout=TIMEOUT)
    if p.is_alive():
        print('*** TIMEOUT for Template Identification')
        p.terminate()
        return
    parse_time = time.time() - start_time  # end_time is the wall-clock time in seconds

    if otc:
        # use a structured log file with corrected oracle templates
        groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')
    else:
        groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')

    parsedresult = os.path.join(output_dir, log_file_basename + '_structured.csv')
    print("parsedresult",parsedresult)
    # calculate grouping accuracy
    start_time = time.time()
    _, GA = evaluate(
        groundtruth=groundtruth,
        parsedresult=parsedresult
    )
    GA_end_time = time.time() - start_time
    print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(GA_end_time))

    # calculate parsing accuracy
    start_time = time.time()
    PA = calculate_parsing_accuracy(
        groundtruth=groundtruth,
        parsedresult=parsedresult
    )
    PA_end_time = time.time() - start_time
    print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    avg_accu, partial,tool_templates, ground_templates, FTA, PTA, RTA, OG, UG, SuccesInd = evaluate_template_level(
        dataset=dataset,
        groundtruth=groundtruth,
        parsedresult=parsedresult,
        output_dir=input_dir+"/output"
    )
    TA_end_time = time.time() - start_time
    print('Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))
    print('Template-level accuracy calculation xxxxxxx', tool_templates, ground_templates)

    result = dataset + ',' + \
             str(tool_templates) + ',' + \
             str(ground_templates) + ',' + \
             str(avg_accu) + ',' + \
             str(GA) + ',' + \
             str(PA) + ',' + '\n'



    
    output_dir=input_dir+"/output"
    #print(output_dir)
    with open(os.path.join(output_dir, result_file+'_ommission.csv'), 'a') as summary_file:
        
        summary_file.write(dataset)
        summary_file.write('\n')
        summary_file.write(str(SuccesInd))
        summary_file.write('\n')
    with open(os.path.join(output_dir, result_file+'_match.csv'), 'a') as summary_file:
        
        summary_file.write(dataset)
        summary_file.write('\n')
        summary_file.write(str(partial))
        summary_file.write('\n')
    
    with open(os.path.join(output_dir, result_file+'_accuracy.csv'), 'a') as summary_file:
        summary_file.write(dataset)
        summary_file.write(result)
        summary_file.write('\n')
