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

from __future__ import print_function

import os
import pandas as pd


from evaluation.utils.common import is_abstract,  transform

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if (value in lst2 and value.isalnum())]
    return lst3

def differ(lst1,lst2):
    return list(set(lst1) - set(lst2))

def to_list(str):
    #lst1 = str.split()
    lst3 = [value for value in str if (value.isalnum())]
    return lst3

def evaluate_template_level(dataset, groundtruth, parsedresult, output_dir):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    oracle_templates = list(pd.read_csv(groundtruth)['EventTemplate'].drop_duplicates().dropna())
    identified_templates = list(pd.read_csv(parsedresult)['EventTemplate'].drop_duplicates().dropna())
    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)
    count_match =0
    total_match =0
    count_partial =0
    count_mismatch =0

    comparison_results = []
    for identified_template in identified_templates:
        identified_template_type = None

        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        log_message_ids_ground = groundtruth_df.loc[groundtruth_df['EventTemplate'] == identified_template, 'LineId']
        num_messages_ground=len(log_message_ids_ground)
        num_messages=len(log_message_ids)
        
        if num_messages == 1:
            continue
        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)
        corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
        
        

        number_corresponding_oracle_templates = len(list(corresponding_oracle_templates.EventTemplate))
        corr= pd.DataFrame(columns=['A'])
        corr['A']=corr_oracle_templates
        parse= pd.DataFrame( columns=['B'])
        parse['B']=[identified_template]
        ground_eventId = groundtruth_df.loc[groundtruth_df['EventTemplate'] == str(corr['A'].tolist()[0])]
        sss= set(ground_eventId.EventId.tolist())
        event_id = list(sss)[0]
        print("event_id", event_id)
        #print(set(corr_oracle_templates))

        # Check SM (SaMe)
        #corr['A'] = corr['A'].map(lambda x:transform(str(x)))

        #parse['B'] = parse['B'].map(lambda x:transform(str(x)))

        A1 = ' '.join(e for e in str(corr['A'].tolist()[0]).lower() if e.isalnum())
        B1=  ' '.join(e for e in str(parse['B'].tolist()[0]).lower()  if e.isalnum())
        lsta= to_list(A1)
        lstb= to_list(B1)
        A = ''.join(e for e in str(corr['A'].tolist()[0]) if e.isalnum())
        B=  ''.join(e for e in str(parse['B'].tolist()[0]) if e.isalnum())
        print("A",lsta)
        print("B",lstb)

        print("difference",differ(lsta,lstb))
        if A.lower()==B.lower():
            identified_template_type = 'Match'
            print("Match")
            count_match +=num_messages
            #total_match +=1

        # incorrect template analysis
        if identified_template_type is None:

            # determine the template type
            template_types = set()
            for corr_oracle_template in corr_oracle_templates:
                #if is_similar(identified_template, corr_oracle_template):
                #if is_abstract(corr_oracle_template, identified_template):
                if len(intersection(lsta,lstb))>= 0.75*len(lsta) :
                    #Under generalization is considered a partial parsing
                    print("partial")
                    template_types.add('Partial')
                    count_partial +=num_messages
                    #total_match +=1
                    #over generalization (removing one static token is not needed)
                else:
                    template_types.add('Mismatch')
                    #count_mismatch +=num_messages
                    #total_match +=1


            if len(template_types) == 1:  # if the set is singleton
                identified_template_type = template_types.pop()


        # save the results for the current identified template
        comparison_results.append([identified_template, identified_template_type, corr_oracle_templates, num_messages,event_id])
    

    comparison_results_df = pd.DataFrame(comparison_results,
                                         columns=['identified_template', 'type', 'corr_oracle_templates', 'num_messages', 'id_ground'])


    partial = comparison_results_df.groupby(['type']).agg({'num_messages':'sum'})
    total = partial['num_messages'].sum()
    #print(total)
    partial = partial.div(total)

    avg_accu = count_match/len(identified_templates)
    #print(avg_accu)
    
    print(comparison_results_df['type'].value_counts())
    
    comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA, OG, UG, MX) = compute_template_level_accuracy(len(oracle_templates), comparison_results_df)

    return avg_accu, partial,len(identified_templates), len(oracle_templates), F1_measure, PTA, RTA, OG, UG, MX


def evaluate_template_level_hdfs(dataset, groundtruth, parsedresult, output_dir):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    oracle_templates = list(pd.read_csv(groundtruth)['EventTemplate'].drop_duplicates().dropna())
    identified_templates = list(pd.read_csv(parsedresult)['EventTemplate'].drop_duplicates().dropna())
    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)
    count_match =0
    total_match =0
    count_partial =0
    count_mismatch =0

    comparison_results = []
    for identified_template in oracle_templates:
        identified_template_type = None
        #print("identified template", identified_template)
    
        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        log_message_ids_ground = groundtruth_df.loc[groundtruth_df['EventTemplate'] == identified_template, 'LineId']
        num_messages_ground=len(log_message_ids_ground)
        num_messages=len(log_message_ids)
        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        num_messages = len(log_message_ids)

        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)


        # Check SM (SaMe)
        if set(corr_oracle_templates) == {identified_template}:
            identified_template_type = 'SM'
        #print("num_messages detected", num_messages)
        #print("num_messages ground",num_messages_ground)
        #check again if unique is really the case, may need to run the experiment match/mismatch again
        if num_messages == 1:
            continue
        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)
        corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
        
        

        number_corresponding_oracle_templates = len(list(corresponding_oracle_templates.EventTemplate))
        corr= pd.DataFrame(columns=['A'])
        corr['A']=corr_oracle_templates
        parse= pd.DataFrame( columns=['B'])
        parse['B']=[identified_template]
        ground_eventId = groundtruth_df.loc[groundtruth_df['EventTemplate'] == identified_template]
        sss= set(ground_eventId.EventId.tolist())
        event_id="NA"
        if len(sss) >0 :
            event_id = list(sss)[0]
        #print("event_id", event_id)
        #print(set(corr_oracle_templates))

        # Check SM (SaMe)
        corr['A'] = corr['A'].map(lambda x:transform(str(x)))

        parse['B'] = parse['B'].map(lambda x:transform(str(x)))
        
        template_ground = corr['A']
        
        #HDFS, HDFS, HDFS
        #concerns only HDFS
        #template_ground= template_ground.replace("blk", "")
        #template_ground= template_ground.replace("/part", " ")
        #template_ground= template_ground.replace("/", " ")
        #template_ground= template_ground.replace("  ", " ")
        
        #print(template_ground)
        #print(str(parse['B'].tolist()[0]).lower())
        
        #if template_ground == str(parse['B'].tolist()[0]).lower():
            #identified_template_type = 'Match'
            #count_match +=1
            #total_match +=1

        # incorrect template analysis
        if identified_template_type is None:

            # determine the template type
            template_types = set()
            for corr_oracle_template in corr_oracle_templates:
                #if is_similar(identified_template, corr_oracle_template):
                if is_abstract(corr_oracle_template, identified_template):
                    #Under generalization is considered a partial parsing
                    template_types.add('Partial')
                    count_partial +=1
                    total_match +=1
                    #over generalization (removing one static token is not needed)
                else:
                    template_types.add('Mismatch')
                    count_mismatch +=1
                    total_match +=1


            if len(template_types) == 1:  # if the set is singleton
                identified_template_type = template_types.pop()


        # save the results for the current identified template
        comparison_results.append([identified_template, identified_template_type, corr_oracle_templates, num_messages,event_id])
    

    comparison_results_df = pd.DataFrame(comparison_results,
                                         columns=['identified_template', 'type', 'corr_oracle_templates', 'num_messages', 'EventId'])
    
    
    grouping_by_eventId = comparison_results_df.groupby(['EventId']).agg({'num_messages':'sum'})

    #print(grouping_by_eventId)
    oracle_templates_frame = pd.read_csv(groundtruth)
    #oracle_ = oracle_templates_frame.groupby(['EventId'])
    ground_data = oracle_templates_frame.groupby(['EventId']).count()
    print("number of templates in the ground data",ground_data)
    #ground_data.apply(print)
    #df3 = pd.merge(ground_data,grouping_by_eventId, on='EventId')
    df3 = pd.concat([ground_data, grouping_by_eventId],axis=1)
    df3['num_messages'] = df3['num_messages'].fillna(0)
    df3 = df3[['EventTemplate','num_messages']]

    df3['ommision'] = (df3['EventTemplate'] - df3['num_messages'])/df3['EventTemplate']
    
    df3['commission'] = (df3['num_messages'] - df3['EventTemplate'])/df3['num_messages']
    df3[df3 < 0] = 0
    df3['ICSI'] = 1 - (df3['ommision']+ df3['commission'])
    if (len(oracle_templates)>=len(identified_templates)):
        std_glob = df3['ICSI'].mean()
    else :
        std_glob = df3['ICSI'].sum() /len(identified_templates)
    print(std_glob)
    
    print(output_dir+"/ommision/LOGRAM")
    df3.to_csv(os.path.join(output_dir+"/ommision/LOGRAM", dataset + '_ommission_results.csv'))
    #print(df3)
    

    print(df3)
    partial = comparison_results_df.groupby(['type']).agg({'num_messages':'sum'})
    total = partial['num_messages'].sum()
    #print(total)
    partial = partial.div(total)

    avg_accu = count_match/len(identified_templates)
    #print(avg_accu)
  
    
    comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA, OG, UG, MX) = compute_template_level_accuracy(len(oracle_templates), comparison_results_df)

    return avg_accu, partial,len(identified_templates), len(oracle_templates), F1_measure, PTA, RTA, OG, UG, std_glob




def find_corr_oracle_templates(log_message_ids, groundtruth_df):
    """
    Identify the corresponding oracle templates for the tool-generated(identified) template

    :param log_message_ids: Log_message ids that corresponds to the tool-generated template
    :param groundtruth_df: Oracle structured file
    :return: Identified oracle templates that corresponds to the tool-generated(identified) template
    """

    corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
    corresponding_oracle_templates = list(corresponding_oracle_templates.EventTemplate.unique())
    return corresponding_oracle_templates


def compute_template_level_accuracy(num_oracle_template, comparison_results_df):
    """Calculate the template-level accuracy values.

    :param num_oracle_template: total number of oracle templates
    :param comparison_results_df: template analysis results (dataFrame)
    :return: f1, precision, recall, over_generalized, under_generalized, mixed
    """
    count_total = float(len(comparison_results_df))
    precision = len(comparison_results_df[comparison_results_df.type == 'SM']) / count_total  # PTA
    recall = len(comparison_results_df[comparison_results_df.type == 'SM']) / float(num_oracle_template)  # RTA
    over_generalized = len(comparison_results_df[comparison_results_df.type == 'OG']) / count_total
    under_generalized = len(comparison_results_df[comparison_results_df.type == 'UG']) / count_total
    mixed = len(comparison_results_df[comparison_results_df.type == 'MX']) / count_total
    f1_measure = 0.0
    if precision != 0 or recall != 0:
        f1_measure = 2 * (precision * recall) / (precision + recall)
    return f1_measure, precision, recall, over_generalized, under_generalized, mixed
