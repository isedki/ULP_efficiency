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
from evaluation.utils.template_level_analysis import  evaluate_template_level, evaluate_template_level2
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
        fw.writerow(['Dataset',  'size','parse_time'])

    return result_file

def prepare_results2(result_file):
    with open(result_file, 'w') as csv_file:
        fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['Dataset', 'size','Parse_time'])
    return result_file

def evaluator(
        dataset,
        size,
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
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    print(indir)
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

    result = dataset + ',' + \
             str(size)+ ',' +\
             str(parse_time)

    
    output_dir=input_dir+"/output"
    #print(output_dir)

    #res = prepare_results2(os.path.join(output_dir, result_file+'_efficiency.csv'))
    with open(result_file, 'a') as summary_file:

        #summary_file.write(dataset)
        summary_file.write(result)
        summary_file.write('\n')

